"""Poincare-ball curvature (c) grid search for MovieLens-1M.

Mirrors the Amazon-side curvature search: same c grid, same three seeds per
point, same selection on validation HR@10 (never test HR, to avoid test-set
leakage).

It lives in this package rather than in ncf/ because it must use this
pipeline's data path, not the reference one:

  - `ncf_negative_sampling_datasets_fast.NCFTrainDatasetFutureBlind` is the
    vectorised version with a dense-user fallback, not the same-named class
    in the reference NCF package.
  - Negatives are RESAMPLED EVERY EPOCH here (`ng_sample()` is called each
    epoch), matching this pipeline's pure-NCF and fast-trainer defaults.
    That differs from the reference Amazon pipeline
    (ncf/ablation_ncf_geometry_vs_optimization.py), which deliberately fixes
    the negatives once for an unrelated reason - train-only versus
    full-history negative exclusion semantics; see that file's build_data()
    docstring. Both are intentional designs, and this script follows the one
    belonging to the pipeline it runs in.
  - It imports `ncf_neumf_on_manifold_architecture_fast`, the variant the
    vectorised trainer uses.

fit_poincare_embeddings below is a byte-for-byte copy of the function of the
same name in ncf/baseline_ncf_poincare_fixed_geometry.py. It is copied rather
than imported so that the two pipelines' same-named modules (each has its own
data-preparation and negative-sampling module) cannot shadow each other
through a shared sys.modules cache. If the c-handling in either copy changes,
update both.

NCF hyperparameters (dropout=0.2, weight_decay=0.01, select_by="hr", a
three-point c grid, three seeds) follow the Amazon curvature-search protocol
rather than this pipeline's raw defaults (dropout=0.0, weight_decay=1e-6):
those settings reduce cross-seed noise enough for a curvature comparison to
mean anything.
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
import time
import copy

import geoopt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_preparation_interactions_fast import (
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
    build_movie_user_matrix,
)
from ncf_negative_sampling_datasets_fast import NCFTrainDatasetFutureBlind, NCFTestDatasetSampled
from vectorized_ranking_evaluation import eval_tensors, evaluate_topk_vec, val_loss_vec
from ncf_neumf_on_manifold_architecture_fast import NeuMFOnManifold

CURVATURE_GRID = [0.1, 0.3, 1.0]


def load_movielens_ratings(movielens_dir):
    ratings_path = os.path.join(movielens_dir, "ratings.dat")
    print(f"Loading ratings from {ratings_path} ...", flush=True)
    df = pd.read_csv(
        ratings_path, sep=":", header=None, engine="c", usecols=[0, 2, 4, 6],
        names=["userId", "movieId", "rating", "timestamp"],
    )
    print(f"Unique users={df['userId'].nunique()} unique items={df['movieId'].nunique()} "
          f"rows={len(df)}", flush=True)
    return df


def build_data(max_users, max_movies, min_seq_len, num_ng, data_root, device):
    ratings_df = load_movielens_ratings(os.path.join(data_root, "ml-1m"))
    df_mapped, _ = prepare_sequences(ratings_df, build_user2seq=False, dedup=True)
    df_sub, user2seq_sub, num_users, num_movies = subsample_users_items(
        df_mapped, max_users=max_users, max_movies=max_movies, min_seq_len=min_seq_len)

    user_pos_set = {u: set(m for (m, ts, r) in seq) for u, seq in user2seq_sub.items()}
    train_events, val_next, test_next = train_val_test_split_next_item(user2seq_sub, min_len=3)

    # neg_exclude="train" (He et al. protocol, the original single-file driver's default) - train-time
    # negatives only avoid TRAIN positives, not val/test (no future-leakage into GI,
    # but also no artificial over-exclusion).
    user_pos_train_set = {u: set() for u in range(num_users)}
    for (u, m) in train_events[:, :2].tolist():
        user_pos_train_set[int(u)].add(int(m))

    user_hist_val_set = user_pos_train_set
    user_hist_test_set = {u: set(items) for u, items in user_pos_train_set.items()}
    for (u, _, val_item) in val_next:
        user_hist_test_set[int(u)].add(int(val_item))

    train_pairs = train_events[:, :2].astype(np.int64)

    train_dataset = NCFTrainDatasetFutureBlind(
        features_pos=train_pairs, num_items=num_movies,
        user_pos_train_set=user_pos_train_set, num_ng=num_ng, seed=42)
    val_dataset = NCFTestDatasetSampled(
        next_triples=val_next, num_items=num_movies,
        user_pos_all_set=user_hist_val_set, num_ng=99, seed=123)
    test_dataset = NCFTestDatasetSampled(
        next_triples=test_next, num_items=num_movies,
        user_pos_all_set=user_hist_test_set, num_ng=99, seed=456)

    val_tensors = eval_tensors(val_dataset, device)
    test_tensors = eval_tensors(test_dataset, device)

    # Item-side raw feature matrix (num_movies x num_users, use_ratings=False -
    # pipeline default), the same matrix the
    # pipeline cdist's into D_input - so the Poincare fit here starts from
    # exactly the same distance information the real pipeline uses.
    movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=False)
    feats = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        D_input_init = torch.cdist(feats, feats)
        D_input_init = D_input_init / D_input_init.max().clamp(min=1e-12)

    return dict(
        num_users=num_users, num_movies=num_movies,
        train_dataset=train_dataset, val_tensors=val_tensors, test_tensors=test_tensors,
        D_input_init=D_input_init,
    )


def fit_poincare_embeddings(D_target, dim, device, epochs=1000, lr=0.01, seed=0, c=1.0):
    """Byte-for-byte copy of ncf/baseline_ncf_poincare_fixed_geometry.py's function
    of the same name - see module docstring for why this isn't imported."""
    torch.manual_seed(seed)
    n = D_target.shape[0]
    manifold = geoopt.PoincareBall(c=c)

    init = torch.randn(n, dim, device=device) * 1e-3
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    iu, ju = torch.triu_indices(n, n, offset=1)
    target = D_target[iu, ju].to(device)
    target = target / target.max() * (3.0 / (c ** 0.5))

    print(f"Poincare MDS: fitting {n} points, dim={dim}, {epochs} epochs...", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        optimizer.zero_grad()
        d_hyp = manifold.dist(z[iu], z[ju])
        loss = ((d_hyp - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if ep % 200 == 0 or ep == epochs - 1:
            print(f"Poincare MDS: epoch {ep}/{epochs} loss={loss.item():.6f} "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)

    with torch.no_grad():
        z_tangent = manifold.logmap0(z)

    return z_tangent.detach()


def train_and_eval_ncf_on_fixed_Z(item_Z, data, device, latent_dim, factor_num=16, num_layers=3,
                                  lr=1e-3, epochs=200, patience=30, seed=0,
                                  dropout=0.2, weight_decay=0.01, num_ng=3,
                                  batch_size=2048, top_k=10, resample_negatives=True):
    """Mirrors GradientIsomapCF_fast.py's "final NCF" block (NeuMFOnManifold
    on a fixed item_Z, per-epoch negative resampling, early-stop by val HR@10)
    almost exactly - the only difference is item_Z comes from a Poincare fit
    here instead of from the GI outer loop."""
    torch.manual_seed(seed)
    item_Z = item_Z.to(device)
    train_dataset = data["train_dataset"]
    vu, vi, vl = data["val_tensors"]

    ncf = NeuMFOnManifold(
        user_num=data["num_users"], latent_dim=latent_dim, factor_num=factor_num,
        num_layers=num_layers, dropout=dropout, model_type="NeuMF-end",
        freeze_item_projection=True, item_projection_init_data=item_Z,
    ).to(device)
    trainable_params = [p for p in ncf.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([num_ng], device=device, dtype=torch.float32)

    best_val_hr, best_state, no_improve = -float("inf"), None, 0
    history = {"val_hr": [], "val_loss": [], "train_loss": []}

    for epoch in range(epochs):
        if resample_negatives:
            train_dataset.ng_sample()
        users_all = train_dataset.users_fill.to(device)
        items_all = train_dataset.items_fill.to(device)
        labels_all = train_dataset.labels_fill.to(device)
        n = users_all.numel()

        ncf.train()
        perm = torch.randperm(n, device=device)
        total_train_loss, n_batches = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            preds = ncf(users_all[idx], items_all[idx], item_Z)
            loss = F.binary_cross_entropy_with_logits(preds, labels_all[idx], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_train_loss / max(1, n_batches)

        ncf.eval()
        score_fn = lambda u, i: ncf(u, i, item_Z)
        with torch.no_grad():
            avg_val_loss = val_loss_vec(score_fn, vu, vi, vl, device, pos_weight=pos_weight)
            val_hr, val_ndcg = evaluate_topk_vec(score_fn, vu, vi, vl, top_k, device)

        history["val_hr"].append(val_hr)
        history["val_loss"].append(avg_val_loss)
        history["train_loss"].append(avg_train_loss)

        if val_hr > best_val_hr:
            best_val_hr, no_improve = val_hr, 0
            best_state = copy.deepcopy(ncf.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        ncf.load_state_dict(best_state)
    ncf.eval()
    tu, ti, tl = data["test_tensors"]
    score_fn = lambda u, i: ncf(u, i, item_Z)
    with torch.no_grad():
        test_hr, test_ndcg = evaluate_topk_vec(score_fn, tu, ti, tl, top_k, device)

    return test_hr, test_ndcg, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--data_root", default=_paths.DATA_DIR)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--mds_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_ng", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    data = build_data(args.max_users, args.max_movies, args.min_seq_len, args.num_ng,
                      args.data_root, device)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    results = []
    for c_val in CURVATURE_GRID:
        print(f"\n{'='*60}\n=== curvature c={c_val} ===\n{'='*60}", flush=True)
        t0 = time.time()
        z_tangent = fit_poincare_embeddings(
            data["D_input_init"], dim=args.latent_dim, device=device,
            epochs=args.mds_epochs, seed=args.seed, c=c_val)
        fit_time = time.time() - t0

        test_hr, test_ndcg, history = train_and_eval_ncf_on_fixed_Z(
            z_tangent, data, device, args.latent_dim, patience=args.patience,
            epochs=args.epochs, dropout=args.dropout, weight_decay=args.weight_decay,
            num_ng=args.num_ng, seed=args.seed)
        best_val_hr = max(history["val_hr"]) if history["val_hr"] else float("nan")

        print(f"c={c_val}: best_val_hr={best_val_hr:.4f}  test_hr={test_hr:.4f} "
              f"test_ndcg={test_ndcg:.4f}  fit_time={fit_time:.1f}s", flush=True)

        results.append({
            "c": c_val, "best_val_hr": best_val_hr, "test_hr": float(test_hr),
            "test_ndcg": float(test_ndcg), "fit_time_s": fit_time,
        })

    best = max(results, key=lambda r: r["best_val_hr"])
    print(f"\n=== CURVATURE SEARCH SUMMARY ml1m ({args.tag}) ===")
    for r in results:
        marker = "  <-- BEST (by val_hr)" if r is best else ""
        print(f"c={r['c']:.2f}: val_hr={r['best_val_hr']:.4f} test_hr={r['test_hr']:.4f} "
              f"test_ndcg={r['test_ndcg']:.4f}{marker}", flush=True)

    out_path = os.path.join(HERE, f"results_poincare_curvature_search_ml1m{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "best_c": best["c"],
                   "max_users": args.max_users, "max_movies": args.max_movies}, f, indent=2)
    print(f"[Save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
