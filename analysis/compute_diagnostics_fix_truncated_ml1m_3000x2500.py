"""Recomputes the three ml1m latent32 3000x2500 diagnostics CSVs.

These were read while the training jobs were still writing snapshots, so they
stop at epoch 161-163 while the runs themselves reach 299 with all 300
snapshots present. The README flags them as known-bad, but no walker ever
replaced them: `compute_diagnostics_remaining_runs.py` skips any run that already has a CSV,
so a truncated file is self-perpetuating. The stale files are moved aside
rather than deleted, so the old numbers stay auditable.

Striding matches `diag_ml1m_and_sasrec.py` exactly - every 20th snapshot plus
the last, staged into a temp directory, because `analyze_run` has no stride
parameter and processes whatever is in the folder it is given. This is not
only about cost. The path-length measures sum |diff| between consecutive rows,
so a CSV with 300 rows and one with 16 are not comparable: more sample points
means a longer measured path for the same trajectory. Every other SASRec CSV
in `diagnostics/` was produced at stride 20, and so must these be.

__main__ guard is mandatory: analyze_run forks workers and Windows children
re-import the module.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import glob
import os
import shutil
import sys
import tempfile
import time

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.ANALYSIS_DIR)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import analyze_run  # noqa: E402

SRC = _paths.SASREC_RUNS_DIR
OUT = _paths.DIAGNOSTICS_DIR
STALE = os.path.join(OUT, "_stale_truncated")
RUNS = [f"sasrec_ml1m_latent32_3000x2500_seed{s}" for s in (0, 1, 2)]
N_WORKERS = 12          # 20 physical cores, nothing else running
STRIDE = 20


def epoch_of(path):
    return int("".join(c for c in os.path.basename(path) if c.isdigit()))


def main():
    os.makedirs(STALE, exist_ok=True)
    for run in RUNS:
        path = os.path.join(SRC, run)
        n = len(glob.glob(os.path.join(path, "matrices_epoch*.npz")))
        fin = len(glob.glob(os.path.join(path, "*.pt")))
        if n < 300 or fin == 0:
            print(f"[SKIP] {run}: {n} snapshots, {fin} models - not finished", flush=True)
            continue

        out_csv = os.path.join(OUT, f"geometry_diagnostics_sasrec_{run}.csv")
        if os.path.exists(out_csv):
            shutil.move(out_csv, os.path.join(STALE, os.path.basename(out_csv)))
            print(f"[moved aside] old CSV -> _stale_truncated/", flush=True)

        snaps = sorted(glob.glob(os.path.join(path, "matrices_epoch*.npz")),
                       key=epoch_of)
        keep = snaps[::STRIDE]
        if snaps[-1] not in keep:
            keep.append(snaps[-1])          # the final epoch always counts

        tmp = tempfile.mkdtemp(prefix=f"diag_{run}_")
        try:
            for sn in keep:
                shutil.copy2(sn, os.path.join(tmp, os.path.basename(sn)))
            init = os.path.join(path, "D_input_init.npy")
            if os.path.exists(init):
                shutil.copy2(init, os.path.join(tmp, "D_input_init.npy"))
            print(f"\n=== {run}: {len(keep)} of {len(snaps)} snapshots "
                  f"(stride {STRIDE}) ===", flush=True)
            t0 = time.time()
            analyze_run(tmp, max_epochs_to_show=None, out_csv=out_csv,
                        n_workers=N_WORKERS)
            print(f"[ok] {run}: {time.time() - t0:.0f}s", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
