"""
Classic/vanilla SASRec baseline (real nn.Embedding, no manifold at all),
parameterized for a fair comparison against the fixed GI-SASRec runs at
all 3 scales tested. Reuses sasrec_classic_architecture_and_data.py's SASRec model class,
TrainDataset/TestDataset, and eval functions directly (these are already
architecturally a clean match for SASRecOnManifold - same attention/FFN/
LayerNorm stack, the only difference is a real trainable item_emb table
instead of a manifold-derived projection) - but sasrec_classic_architecture_and_data.py's
own main() hardcodes max_users=300/max_movies=800/maxlen=50/dropout=0.2
and isn't parameterized by scale, so this script re-implements main()'s
training loop with real parameters instead of calling it directly.

Config matched to GI-SASRec: maxlen=30, dropout_rate=0.0, weight_decay=0.0
(added - the original had none), epochs=50/patience=30 (matching
final_cf_epochs/final_patience), batch_size=64 (already matched).
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sasrec_classic_architecture_and_data import SASRec, TrainDataset, TestDataset, load_amazon_data, evaluate_model, evaluate_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon", choices=["amazon", "movielens"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--maxlen", type=int, default=30)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    CONFIG = {
        'hidden_units': args.hidden_units, 'maxlen': args.maxlen,
        'num_blocks': args.num_blocks, 'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
    }

    train_seqs, val_targets, test_targets, num_users, num_items, user_pos_set = load_amazon_data(
        dataset_dir_name=args.dataset_dir_name, category=args.amazon_category,
        max_users=args.max_users, max_movies=args.max_movies, min_seq_len=5,
        dataset_type=args.dataset_type)

    train_dataset = TrainDataset(train_seqs, num_items, CONFIG['maxlen'])
    val_dataset = TestDataset(train_seqs, val_targets, num_items, CONFIG['maxlen'], user_pos_set)
    test_seqs_for_eval = {u: seq + [val_targets[u]] for u, seq in train_seqs.items() if u in val_targets}
    test_dataset = TestDataset(test_seqs_for_eval, test_targets, num_items, CONFIG['maxlen'], user_pos_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SASRec(CONFIG, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98),
                                  weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training classic SASRec: num_users={num_users} num_items={num_items} "
          f"maxlen={args.maxlen} dropout={args.dropout_rate} weight_decay={args.weight_decay}", flush=True)

    best_val_hr = -1.0
    best_model_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        for seq, pos, neg in train_loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(seq, pos, neg)
            indices = torch.where(pos != num_items)
            loss = criterion(pos_logits[indices], torch.ones_like(pos_logits[indices]))
            loss += criterion(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)

        val_hr, val_ndcg = evaluate_model(model, val_loader, device, top_k=10)
        val_loss = evaluate_loss(model, val_loader, device, num_items, criterion)
        print(f"  ep {epoch+1}/{args.epochs} | train={avg_loss:.4f} val={val_loss:.4f} "
              f"val_hr@10={val_hr:.4f} val_ndcg@10={val_ndcg:.4f}", flush=True)

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"  early stop at ep {epoch+1}", flush=True)
            break

    model.load_state_dict(best_model_state)
    test_hr, test_ndcg = evaluate_model(model, test_loader, device, top_k=10)
    elapsed = time.time() - t0

    print(f"\n=== CLASSIC SASREC BASELINE SUMMARY ({args.tag}) ===")
    print(f"best_val_hr={best_val_hr:.4f} test_hr={test_hr:.4f} test_ndcg={test_ndcg:.4f} "
          f"elapsed={elapsed:.1f}s", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(_paths.SASREC_RUNS_ROOT, f"results_classic_sasrec{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"test_hr": float(test_hr), "test_ndcg": float(test_ndcg),
                   "best_val_hr": float(best_val_hr), "elapsed_s": elapsed,
                   "max_users": args.max_users, "max_movies": args.max_movies,
                   "maxlen": args.maxlen, "seed": args.seed}, f, indent=2)
    print(f"[Save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
