"""Saves the item geometry of classic SASRec, in the configuration the results
table actually uses.

Why not `the superseded SASRec Euclidean dump`: that script trains at
`maxlen=50, dropout=0.2` and writes to a single hardcoded filename. Both are
problems here. The configuration does not match the classic-SASRec arm in the
table (`baseline_sasrec_classic_euclidean.py`: maxlen=30, dropout=0.0), so its saved
geometry belongs to a different model than the numbers it would be captioned
with - measured: on Grocery 1000x1200 the stored geometry scored 0.3535 while
the table's classic arm scores 0.4101. And the fixed filename means a sweep
over datasets silently overwrites itself.

This file reuses `run_sasrec_classic_baseline`'s own training path so the
geometry and the table come from the same model, and writes one tagged npz per
configuration carrying D, the test metrics and the config that produced them.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import copy  # noqa: E402
import time  # noqa: E402

import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sasrec_classic_architecture_and_data import (  # noqa: E402
    SASRec, TrainDataset, TestDataset, load_amazon_data, evaluate_model, evaluate_loss,
)


def train_classic(model, train_loader, val_loader, device, cfg, num_items):
    """Same loop as run_sasrec_classic_baseline.main: sampled BCE on
    pos/neg pairs, early stopping on validation HR@10, best checkpoint
    restored. Kept here verbatim because that script exposes no reusable
    function - it is one monolithic main()."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           betas=(0.9, 0.98), weight_decay=0.0)
    crit = nn.BCEWithLogitsLoss()
    best_hr, best_state, no_improve = -1.0, None, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for seq, pos, neg in train_loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(seq, pos, neg)
            idx = torch.where(pos != num_items)
            loss = crit(pos_logits[idx], torch.ones_like(pos_logits[idx]))
            loss = loss + crit(neg_logits[idx], torch.zeros_like(neg_logits[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        val_hr, _ = evaluate_model(model, val_loader, device, top_k=10)
        print(f"  ep {ep+1}/{cfg['epochs']} train={tot/max(1,nb):.4f} "
              f"val_hr@10={val_hr:.4f}", flush=True)
        if val_hr > best_hr:
            best_hr, best_state, no_improve = val_hr, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  early stop at ep {ep+1}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  training took {time.time()-t0:.0f}s, best val HR@10={best_hr:.4f}", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_users", type=int, default=300)
    p.add_argument("--max_movies", type=int, default=800)
    p.add_argument("--dataset_dir_name", default="amazon_beauty")
    p.add_argument("--dataset_type", default="amazon", choices=["amazon", "movielens"])
    p.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    p.add_argument("--maxlen", type=int, default=30)
    p.add_argument("--hidden_units", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", default=os.path.join(_paths.SASREC_RUNS_ROOT, "euclidean_geometry_aligned"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    CONFIG = {"hidden_units": args.hidden_units, "maxlen": args.maxlen,
              "num_blocks": args.num_blocks, "num_heads": args.num_heads,
              "dropout_rate": args.dropout_rate, "learning_rate": 1e-3,
              "batch_size": args.batch_size, "epochs": args.epochs,
              "patience": args.patience}

    train_seqs, val_t, test_t, num_users, num_items, user_pos = load_amazon_data(
        max_users=args.max_users, max_movies=args.max_movies,
        dataset_dir_name=args.dataset_dir_name, category=args.amazon_category,
        dataset_type=args.dataset_type)
    print(f"users={num_users} items={num_items} maxlen={args.maxlen} "
          f"dropout={args.dropout_rate}", flush=True)

    train_ds = TrainDataset(train_seqs, num_items, CONFIG["maxlen"])
    val_ds = TestDataset(train_seqs, val_t, num_items, CONFIG["maxlen"], user_pos)
    test_seqs = {u: s + [val_t[u]] for u, s in train_seqs.items() if u in val_t}
    test_ds = TestDataset(test_seqs, test_t, num_items, CONFIG["maxlen"], user_pos)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = SASRec(CONFIG, num_items).to(device)
    model = train_classic(model, train_loader, val_loader, device, CONFIG, num_items)
    hr, ndcg = evaluate_model(model, test_loader, device, top_k=10)
    print(f"[Test] HR@10={hr:.4f} NDCG@10={ndcg:.4f}", flush=True)

    emb = model.item_emb.weight[:-1].detach().cpu().numpy().astype(np.float32)
    D = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1).astype(np.float64)
    out = os.path.join(args.out_dir, f"euclidean_geometry_{args.tag}.npz")
    np.savez(out, D=D, Z=emb, test_hr=float(hr), test_ndcg=float(ndcg),
             maxlen=args.maxlen, dropout=args.dropout_rate, seed=args.seed)
    with open(os.path.join(args.out_dir, f"meta_{args.tag}.json"), "w") as f:
        json.dump({"test_hr": float(hr), "test_ndcg": float(ndcg), "config": CONFIG,
                   "dataset": args.dataset_dir_name, "scale": f"{args.max_users}x{args.max_movies}",
                   "seed": args.seed}, f, indent=2)
    print(f"[Save] {out}", flush=True)


if __name__ == "__main__":
    main()
