"""
Curvature c for the HypSASRec arm, computed exactly as the authors do.

Reference: Hyperbolic-SASRec (Frolov et al.), scripts/compute_curvature.py and
lib/delta.py. Their procedure:

  1. item representations = truncated SVD of the training interaction matrix,
     coordinates taken as (singular vectors * singular values), rank = the
     model's embedding dimension;
  2. relative delta-hyperbolicity delta_rel = 2 * delta / diam, estimated on
     random batches of items (n_tries=10, batch_size=1500);
  3. c = (delta_rel(Poincare disk) / mean(delta_rel(X)))**2, with the Poincare
     anchor evaluated at tol=1e-12 -- the tolerance recorded in every one of
     the authors' own best configs ("obtained from SVD embeddings at 1e^{-12}
     tolerance").

delta itself is computed with THIS project's own exact 4-point implementation
(analyze_hyperbolicity.delta_hyperbolicity) rather than the authors' numba
kernel: numba is not installed in the active venv and installing into it while
three training jobs are running is not worth the risk. The quantity is the
same -- exact Gromov delta via the basepoint trick, then 2*delta/diam.

Data pipeline is the project's own (prepare_sequences / subsample_users_items /
train_val_test_split_next_item) so that the curvature is computed on exactly
the subsample each run trains on. Writes a JSON next to itself; touches nothing
in the repository.
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

SCALES = [(300, 800), (1000, 1200), (3000, 2500)]
SVD_RANK = 32          # == hidden_units of the HypSASRec arm
N_TRIES = 10           # authors' batched_delta_hyp default
BATCH_SIZE = 1500      # authors' batched_delta_hyp default
MIN_SEQ_LEN = 5        # the SASRec family's subsampling filter
SEED = 42


def relative_delta_poincare(tol=1e-12):
    """Verbatim from the authors' lib/delta.py (their default is 1e-5; every
    one of their best configs records 1e-12, which is what we use)."""
    r = 1.0 - tol
    return np.log1p(np.sqrt(2)) / np.log1p(2 * r / (1 - r))


def item_representations(train_events, num_items, num_users, rank, binary=True):
    """Authors' get_item_representations, adapted to this project's data.

    They SVD the (users x items) training matrix and take the right singular
    vectors scaled by singular values. Building the transpose (items x users)
    and taking the LEFT singular vectors scaled the same way is identical.
    """
    mat = np.zeros((num_items, num_users), dtype=np.float32)
    for u, i, r in train_events:
        if binary:
            mat[int(i), int(u)] = 1.0
        else:
            mat[int(i), int(u)] = float(r)
    k = min(rank, min(mat.shape))
    U, S, _ = np.linalg.svd(mat, full_matrices=False)
    order = np.argsort(S)[::-1][:k]
    return np.ascontiguousarray(U[:, order] * S[order]), mat


def batched_delta_rel(X, n_tries=N_TRIES, batch_size=BATCH_SIZE, seed=SEED):
    """Authors' batched_delta_hyp: random item subsets, delta_rel per subset."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n <= batch_size:
        n_tries = 1  # every "batch" would be the identical full set
    deltas = []
    for t in range(n_tries):
        idx = np.arange(n) if n <= batch_size else rng.choice(n, batch_size, replace=False)
        cond = pdist(X[idx].astype(np.float64))
        diam = float(cond.max())
        delta = delta_hyperbolicity(squareform(cond), basepoint=0)
        deltas.append(2.0 * delta / diam if diam > 0 else float("nan"))
        print(f"      try {t + 1}/{n_tries}: n={len(idx)} diam={diam:.4f} "
              f"delta_rel={deltas[-1]:.4f}", flush=True)
    return deltas


def main():
    anchor = relative_delta_poincare(1e-12)
    print(f"Poincare anchor delta_rel(P) at tol=1e-12: {anchor:.6f}\n", flush=True)

    datasets = [
        ("ml1m", "movielens", os.path.join(DATA, "ml-1m")),
        ("amazon_beauty", "amazon", os.path.join(DATA, "amazon_beauty")),
    ]

    results = []
    for name, kind, path in datasets:
        print(f"=== {name}: loading ===", flush=True)
        t0 = time.time()
        if kind == "movielens":
            df = pd.read_csv(os.path.join(path, "ratings.dat"), sep="::", engine="python",
                             header=None, names=["userId", "movieId", "rating", "timestamp"])
        else:
            df = pd.read_csv(os.path.join(path, "Beauty_and_Personal_Care.csv"))
            df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
            df = df[["userId", "movieId", "rating", "timestamp"]]
        print(f"    loaded {len(df)} ratings in {time.time() - t0:.0f}s", flush=True)
        df_mapped, _ = prepare_sequences(df)

        for (N, M) in SCALES:
            print(f"  --- {name} {N}x{M} ---", flush=True)
            _, u2s, num_users, num_items = subsample_users_items(
                df_mapped.copy(), max_users=N, max_movies=M, min_seq_len=MIN_SEQ_LEN)
            train_events, _, _ = train_val_test_split_next_item(u2s, min_len=3)

            row = {"dataset": name, "scale": f"{N}x{M}",
                   "num_users": int(num_users), "num_items": int(num_items),
                   "svd_rank": SVD_RANK, "tol": 1e-12, "anchor_delta_rel_poincare": anchor}

            for binary in (True, False):
                tag = "binary" if binary else "ratings"
                X, _ = item_representations(train_events, num_items, num_users,
                                            SVD_RANK, binary=binary)
                print(f"    [{tag}] SVD embeddings {X.shape}", flush=True)
                deltas = batched_delta_rel(X)
                mean_delta = float(np.mean(deltas))
                c = float((anchor / mean_delta) ** 2)
                row[f"delta_rel_mean_{tag}"] = mean_delta
                row[f"delta_rel_std_{tag}"] = float(np.std(deltas))
                row[f"c_{tag}"] = c
                print(f"    [{tag}] mean delta_rel={mean_delta:.4f} -> c={c:.6f}", flush=True)

            results.append(row)
            with open(os.path.join(HERE, "hypsasrec_curvature.json"), "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== SUMMARY (c for the HypSASRec arm) ===")
    print(f"{'dataset':16s} {'scale':12s} {'items':>6s} {'c (binary)':>12s} {'c (ratings)':>12s}")
    for r in results:
        print(f"{r['dataset']:16s} {r['scale']:12s} {r['num_items']:6d} "
              f"{r['c_binary']:12.6f} {r['c_ratings']:12.6f}")


if __name__ == "__main__":
    main()
