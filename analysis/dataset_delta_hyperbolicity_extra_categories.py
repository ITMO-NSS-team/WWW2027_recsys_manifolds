"""
Relative delta-hyperbolicity of the three additional Amazon categories
(Arts & Crafts, Grocery & Gourmet Food, Office Products), computed with
exactly the procedure already used for MovieLens-1M and Amazon Beauty:

  item representations = rank-32 truncated SVD of the binary training
  interaction matrix (authors' get_item_representations), delta_rel =
  2*delta/diam estimated over random item batches (n_tries=10,
  batch_size=1500), c = (delta_rel(Poincare @ tol=1e-12) / mean delta_rel)^2.

Subsampling uses this project's own prepare_data pipeline with the SASRec
family's min_seq_len=5, so the numbers describe exactly the subsamples the
models were trained on. Read-only; writes one JSON next to itself.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)
sys.path.insert(0, _paths.ANALYSIS_DIR)

from data_preparation_interactions import (  # noqa: E402
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
)
from gromov_delta_hyperbolicity import delta_hyperbolicity  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _paths.DATA_DIR

DATASETS = [
    ("arts", "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing"),
    ("grocery", "amazon_grocery_gourmet_food", "Grocery_and_Gourmet_Food"),
    ("office", "amazon_office_products", "Office_Products"),
]
SCALES = [(300, 800), (1000, 1200), (3000, 2500)]
SVD_RANK, N_TRIES, BATCH, MIN_SEQ_LEN, SEED = 32, 10, 1500, 5, 42


def relative_delta_poincare(tol=1e-12):
    r = 1.0 - tol
    return np.log1p(np.sqrt(2)) / np.log1p(2 * r / (1 - r))


def item_representations(train_events, num_items, num_users, rank):
    mat = np.zeros((num_items, num_users), dtype=np.float32)
    for u, i, _ in train_events:
        mat[int(i), int(u)] = 1.0
    U, S, _ = np.linalg.svd(mat, full_matrices=False)
    k = min(rank, len(S))
    order = np.argsort(S)[::-1][:k]
    return np.ascontiguousarray(U[:, order] * S[order])


def batched_delta_rel(X, seed=SEED):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    tries = 1 if n <= BATCH else N_TRIES
    out = []
    for t in range(tries):
        idx = np.arange(n) if n <= BATCH else rng.choice(n, BATCH, replace=False)
        cond = pdist(X[idx].astype(np.float64))
        diam = float(cond.max())
        delta = delta_hyperbolicity(squareform(cond), basepoint=0)
        out.append(2.0 * delta / diam if diam > 0 else float("nan"))
        print(f"      try {t + 1}/{tries}: n={len(idx)} delta_rel={out[-1]:.4f}", flush=True)
    return out


def main():
    anchor = relative_delta_poincare(1e-12)
    print(f"anchor delta_rel(P) = {anchor:.6f}\n", flush=True)
    results = []
    for name, dirname, category in DATASETS:
        path = os.path.join(DATA, dirname, f"{category}.csv")
        print(f"=== {name}: loading {os.path.basename(path)} ===", flush=True)
        t0 = time.time()
        df = pd.read_csv(path).rename(columns={"user_id": "userId", "parent_asin": "movieId"})
        df = df[["userId", "movieId", "rating", "timestamp"]]
        print(f"    {len(df)} ratings in {time.time() - t0:.0f}s", flush=True)
        df_mapped, _ = prepare_sequences(df)

        for (N, M) in SCALES:
            print(f"  --- {name} {N}x{M} ---", flush=True)
            _, u2s, num_users, num_items = subsample_users_items(
                df_mapped.copy(), max_users=N, max_movies=M, min_seq_len=MIN_SEQ_LEN)
            train_events, _, _ = train_val_test_split_next_item(u2s, min_len=3)
            X = item_representations(train_events, num_items, num_users, SVD_RANK)
            deltas = batched_delta_rel(X)
            mean_d = float(np.mean(deltas))
            row = {"dataset": name, "scale": f"{N}x{M}", "num_users": int(num_users),
                   "num_items": int(num_items), "delta_rel_mean": mean_d,
                   "delta_rel_std": float(np.std(deltas)), "c": float((anchor / mean_d) ** 2)}
            print(f"    delta_rel={mean_d:.4f}  c={row['c']:.6f}", flush=True)
            results.append(row)
            with open(os.path.join(HERE, "delta_new_datasets.json"), "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'dataset':10s} {'scale':12s} {'items':>6s} {'delta_rel':>10s} {'c':>10s}")
    for r in results:
        print(f"{r['dataset']:10s} {r['scale']:12s} {r['num_items']:6d} "
              f"{r['delta_rel_mean']:10.4f} {r['c']:10.6f}")


if __name__ == "__main__":
    main()
