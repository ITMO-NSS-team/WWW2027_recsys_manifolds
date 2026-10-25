"""
Geometry/topology diagnostics (BEFORE vs AFTER) for the non-parametric
("np") recommendation heads - ItemKNN and bary_cos (Rocchio-style
barycenter-cosine) - layered on gradient-isomap-optimized ml-1m geometry.

Scope: the gradient-Isomap ml-1m grid only, six cells. The per-cell
Ollivier-Ricci computation at 3000x2500 costs several minutes per
BEFORE/AFTER snapshot (all-pairs Dijkstra plus a per-edge W1 solve on a
~2500-node, ~75-average-degree graph), which makes wider coverage expensive.
The Poincare variants on Amazon and ml-1m (twelve further cells) are NOT
computed here - do not report these results as full coverage.

These heads have trainable_head_params = 0: they cannot "launder" a bad
geometry into a useful representation the way NCF's item-projection layers
can. The question this script answers: does the underlying gradient-isomap
geometry have to deform MORE than it does for NCF/SASRec in order for these
simple, non-parametric heads to gain anything - i.e. is |orc_mean delta| /
|spectral_lambda2 delta| for these heads LARGER than NCF's (-0.331 / -0.112)
and SASRec's (-0.041 / -0.047) on the same ml-1m 1000x1200 data?

One cell per (head x scale) = 6 cells: bary_cos and item_knn, each at
300x800 / 1000x1200 / 3000x2500. For each cell the WINNING hyperparameter
combo is picked by mean-of-3-seeds test HR@10 (computed from the
pre-extracted inventory CSV the inventory CSV), then seed=0
of that winning combo is the run actually analysed.

FORMAT DECISION (read before trusting the BEFORE numbers):
The brief's literal instruction is BEFORE = D_input_init.npy, AFTER =
best/matrices_epoch<N>.npz's D_input. D_input_init.npy is a bare (N,N)
array with no knn_adj/D_geodesic/D_latent alongside it, so orc_mean and
spectral_lambda2 (which need a kNN graph) and spearman_rho / h0 persistence
(which need a geodesic/latent pair) are not directly computable from it in
isolation. matrices_epoch0.npz, which IS saved for every one of these runs,
contains a self-consistent (D_input, D_geodesic, knn_adj, Z, D_latent)
snapshot built with the exact same _build_knn_adjacency(D_input, k) call
gradient_isomap_ncf_bilevel_trainer.py uses for every later epoch - and spot-checking one
run (bary_cos 300x800, winning combo) showed epoch0's D_input differs from
D_input_init.npy by at most 0.001 (D_input is normalised to [0, 1]), i.e.
they are the same state up to one optimizer step. So: BEFORE =
matrices_epoch0.npz (full snapshot), AFTER = best/matrices_epoch<N>.npz
(full snapshot) - this is also exactly how the existing NCF/SASRec
epoch0-vs-final comparison elsewhere is computed, which is what we need
to be comparable against.

Reused as-is from geometry_topology_diagnostics.py: ollivier_ricci_curvature (via
summarize_epoch), graph_laplacian_spectral, isometric_embedding_quality
(-> spearman_rho), persistent_homology (-> geo/lat h0_max_persistence).
Kruskal stress and H1 are computed incidentally by summarize_epoch() but
are NOT reported per the "trustworthy measures" table (skip, unreliable /
expensive).

Nothing is written to EXTERNAL_RUNS_DIR - it is read-only here.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# The analysis package (this script's logical home) holds
# geometry_topology_diagnostics.py; this file itself lives in an agent worktree that
# doesn't carry recsys/ at all, so point sys.path at the real location
# explicitly rather than relying on __file__.
ANALYSIS_DIR = _paths.ANALYSIS_DIR
sys.path.insert(0, ANALYSIS_DIR)
from geometry_topology_diagnostics import summarize_epoch, _progress  # noqa: E402

SHARE_ROOT = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baselines_ml-1m")
OUT_DIR = _paths.TABLES_DIR
OUT_CSV = os.path.join(_paths.TABLES_DIR, "diag_np_baselines_results.csv")
SEEDS_OUT_CSV = os.path.join(_paths.TABLES_DIR, "diag_np_baselines_seeds_results.csv")

# Winning hyperparameter combo per (head, scale), picked by mean-of-3-seeds
# test HR@10 from the inventory CSV (see report for the
# full per-combo mean table). seed_0 of each winning combo is analysed by
# default; run_all_seeds() below reruns the SAME combo at seed 1 and 2 too.
CELLS = [
    # head,       scale,         subdir (relative to SHARE_ROOT/<head>/)
    ("bary_cos",  "300x800",     r"lr_1e-2\n_neighbors_10\300_800_seed_0"),
    ("bary_cos",  "1000x1200",   r"lr_1e-2\n_neighbors_10\1000_1200_seed_0"),
    ("bary_cos",  "3000x2500",   r"lr_1e-2\n_neighbors_40\3000_2500_seed_0"),
    ("item_knn",  "300x800",     r"nn_10_lr_1e-3\knn_k_20_temp_0.5\300_800_seed_0"),
    ("item_knn",  "1000x1200",   r"nn_10_lr_1e-3\knn_k_20_temp_0.5\1000_1200_seed_0"),
    ("item_knn",  "3000x2500",   r"nn_40_lr_1e-2\knn_k_20_temp_0.5\3000_2500_seed_0"),
]

MEASURES = ["orc_mean", "spectral_lambda2", "spearman_rho",
            "geo_h0_max_persistence", "lat_h0_max_persistence"]


def analyze_cell(head: str, scale: str, subdir: str) -> dict:
    run_dir = os.path.join(SHARE_ROOT, head, subdir)
    _progress(f"=== {head} {scale} : {run_dir} ===")

    before_path = os.path.join(run_dir, "matrices_epoch0.npz")
    best_files = glob.glob(os.path.join(run_dir, "best", "matrices_epoch*.npz"))
    assert len(best_files) == 1, f"expected exactly 1 best-epoch file, got {best_files}"
    after_path = best_files[0]

    # Sanity check the format decision: D_input_init.npy vs epoch0's D_input.
    d_input_init = np.load(os.path.join(run_dir, "D_input_init.npy"))
    with np.load(before_path) as e0:
        max_diff = float(np.max(np.abs(d_input_init - e0["D_input"])))
    _progress(f"D_input_init vs epoch0 D_input: max abs diff = {max_diff:.6f}")

    with open(os.path.join(run_dir, "run_log.json")) as f:
        run_log = json.load(f)
    hr = run_log.get("summary", {}).get("geometry_baseline_test", {}).get("hr")
    ndcg = run_log.get("summary", {}).get("geometry_baseline_test", {}).get("ndcg")

    t0 = time.time()
    with np.load(before_path) as d:
        before = summarize_epoch(d)
    _progress(f"before done ({time.time() - t0:.0f}s)")

    t0 = time.time()
    with np.load(after_path) as d:
        after = summarize_epoch(d)
    _progress(f"after done ({time.time() - t0:.0f}s)")

    row = {
        "head": head, "scale": scale, "seed": int(subdir.rsplit("_seed_", 1)[1]),
        "hr_test": hr, "ndcg_test": ndcg,
        "run_dir": run_dir, "after_epoch": os.path.basename(after_path),
        "d_input_init_vs_epoch0_maxdiff": max_diff,
    }
    for m in MEASURES:
        b, a = before.get(m), after.get(m)
        row[f"{m}_before"] = b
        row[f"{m}_after"] = a
        row[f"{m}_delta"] = (a - b) if (b is not None and a is not None) else None
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for head, scale, subdir in CELLS:
        row = analyze_cell(head, scale, subdir)
        results.append(row)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)  # incremental save
        _progress(f"[saved] {OUT_CSV} ({len(results)}/{len(CELLS)})")

    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    cols = ["head", "scale", "hr_test"] + [f"{m}_delta" for m in MEASURES]
    print(df[cols].to_string(index=False))
    print(f"\n[Save] {OUT_CSV}")


def run_all_seeds():
    """Extend the single-seed (seed_0) results above to all 3 seeds, for
    the SAME winning hyperparameter combo per cell (just substituting
    _seed_0 -> _seed_1 / _seed_2 in the subdir string) - 18 runs total.

    Reuses the already-computed seed_0 rows from OUT_CSV (identical
    before/after snapshots, identical diagnostics code - recomputing them
    would just burn the same ~minutes-per-3000x2500-cell ORC cost for a
    bit-identical answer) and only freshly computes seed 1 and seed 2
    (12 new runs)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    cached = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    results = []
    for head, scale, subdir in CELLS:
        base = subdir.rsplit("_seed_", 1)[0]
        # NOTE: must use cached["head"]/cached["scale"], not attribute access -
        # cached.head silently resolves to the DataFrame.head() METHOD (the
        # column name collides with a built-in pandas method), which made
        # this filter match 0 rows every time and silently recompute
        # everything instead of reusing seed_0 - caught only after the fact.
        cached_row = cached[(cached["head"] == head) & (cached["scale"] == scale)] if len(cached) else cached
        if len(cached_row):
            row0 = cached_row.iloc[0].to_dict()
            row0["seed"] = 0
            _progress(f"[reused cached seed_0] {head} {scale}")
        else:
            row0 = analyze_cell(head, scale, f"{base}_seed_0")
        results.append(row0)
        pd.DataFrame(results).to_csv(SEEDS_OUT_CSV, index=False)

        for seed in (1, 2):
            row = analyze_cell(head, scale, f"{base}_seed_{seed}")
            results.append(row)
            pd.DataFrame(results).to_csv(SEEDS_OUT_CSV, index=False)  # incremental save
            _progress(f"[saved] {SEEDS_OUT_CSV} ({len(results)}/18)")

    df = pd.DataFrame(results)
    print("\n=== PER-SEED RESULTS ===")
    cols = ["head", "scale", "seed", "hr_test"] + [f"{m}_delta" for m in MEASURES]
    print(df[cols].to_string(index=False))

    print("\n=== MEAN +/- STD ACROSS 3 SEEDS ===")
    agg_cols = [f"{m}_delta" for m in ["orc_mean", "spectral_lambda2", "spearman_rho"]]
    agg = df.groupby(["head", "scale"])[agg_cols].agg(["mean", "std"])
    print(agg.to_string())
    print(f"\n[Save] {SEEDS_OUT_CSV}")


if __name__ == "__main__":
    run_all_seeds()
