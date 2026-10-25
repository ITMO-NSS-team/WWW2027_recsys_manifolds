"""
Fills the three gaps in the paper-configuration diagnostics set.

The paper's config (eta=0.001, frozen projection) has 12 runs across four
scales, but only 9 carry a geometry_diagnostics.csv; these three were never
analysed even though their matrices_epoch*.npz snapshots are on disk:

    frozenproj_scaletest_1000x1200_seed2_0.001   (300 snapshots)
    frozenproj_scaletest_3000x2500_seed1_0.001   (100 snapshots)
    frozenproj_scaletest_3000x2500_seed2_0.001   (100 snapshots)

Pure offline recomputation from saved matrices - no training, no GPU.
analyze_run writes the CSV incrementally, so an interruption only loses the
snapshots not yet processed.

Guarded with __main__ because analyze_run forks workers: on Windows the
children re-import this module, and without the guard that is a fork bomb
(the same failure already documented for the run_*.py drivers).
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os
import sys
import time

MANUL = _paths.REPO_ROOT
VERIF = _paths.ANALYSIS_DIR
sys.path.insert(0, VERIF)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import analyze_run  # noqa: E402

LOGS = os.path.join(VERIF, "logs_movielens_isomap_cf")
TARGETS = [
    "amazon_beauty_frozenproj_scaletest_1000x1200_seed2_0.001",
    "amazon_beauty_frozenproj_scaletest_3000x2500_seed1_0.001",
    "amazon_beauty_frozenproj_scaletest_3000x2500_seed2_0.001",
]
N_WORKERS = 3   # CPU-only; leaves cores for the GPU jobs' own data loading


def main():
    for name in TARGETS:
        folder = os.path.join(LOGS, name)
        out_csv = os.path.join(folder, "geometry_diagnostics.csv")
        if not os.path.isdir(folder):
            print(f"[skip] no directory: {name}", flush=True)
            continue
        if os.path.exists(out_csv):
            print(f"[skip] already computed: {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            analyze_run(folder, max_epochs_to_show=None, out_csv=out_csv, n_workers=N_WORKERS)
            print(f"[ok] {name}: {time.time() - t0:.0f}s -> {out_csv}", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for name in TARGETS:
        p = os.path.join(LOGS, name, "geometry_diagnostics.csv")
        if os.path.exists(p):
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {name}: {n} epochs", flush=True)
        else:
            print(f"  {name}: MISSING", flush=True)


if __name__ == "__main__":
    main()
