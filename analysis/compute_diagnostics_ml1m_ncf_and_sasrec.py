"""Geometry and topology diagnostics for the SASRec runs on Amazon Beauty
and ML-1M.

These runs save 300 snapshots each, and Ollivier-Ricci curvature on a
2500-item kNN graph is the expensive part, so only every STRIDE-th epoch is
analysed - enough to trace a trajectory, at a fraction of the cost. Runs whose
training job is still writing are skipped by name.

Offline recomputation from saved matrices: no training, no GPU. Writes one CSV
per run plus a manifest of what was produced.

The GI-NCF side of ML-1M is handled by
compute_diagnostics_ncf_paper_config.py, which reads the paper-configuration
tree. (An earlier revision of this script also walked an lr_isomap=0.05
hyperparameter-search tree; that tree is not used anywhere in this
repository.)
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
import re
import shutil
import sys
import tempfile
import time

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.ANALYSIS_DIR)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import analyze_run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _paths.DIAGNOSTICS_DIR
os.makedirs(OUT, exist_ok=True)

SASREC_BASE = _paths.SASREC_RUNS_DIR
SASREC_SKIP = "ml1m_3000x2500"      # still being written by the live jobs
STRIDE = 20
N_WORKERS = 6


def epoch_of(p):
    return int(re.search(r"epoch(\d+)", os.path.basename(p)).group(1))


def analyse(folder, tag, stride=1):
    """analyze_run processes every snapshot in a folder, so when striding we
    stage the wanted snapshots as symlinks/copies in a temp dir."""
    out_csv = os.path.join(OUT, f"geometry_diagnostics_{tag}.csv")
    if os.path.exists(out_csv):
        print(f"[skip] already present: {tag}", flush=True)
        return
    snaps = sorted(glob.glob(os.path.join(folder, "matrices_epoch*.npz")), key=epoch_of)
    if not snaps:
        print(f"[skip] no snapshots: {tag}", flush=True)
        return
    t0 = time.time()
    if stride > 1:
        keep = snaps[::stride]
        if snaps[-1] not in keep:
            keep.append(snaps[-1])          # always include the final epoch
        tmp = tempfile.mkdtemp(prefix=f"diag_{tag}_")
        try:
            for s in keep:
                shutil.copy2(s, os.path.join(tmp, os.path.basename(s)))
            init = os.path.join(folder, "D_input_init.npy")
            if os.path.exists(init):
                shutil.copy2(init, os.path.join(tmp, "D_input_init.npy"))
            print(f"=== {tag}: {len(keep)} of {len(snaps)} snapshots (stride {stride}) ===", flush=True)
            analyze_run(tmp, max_epochs_to_show=None, out_csv=out_csv, n_workers=N_WORKERS)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"=== {tag}: {len(snaps)} snapshots ===", flush=True)
        analyze_run(folder, max_epochs_to_show=None, out_csv=out_csv, n_workers=N_WORKERS)
    print(f"[ok] {tag}: {time.time() - t0:.0f}s", flush=True)


def main():
    produced = []

    print("\n########## SASRec ##########", flush=True)
    if os.path.isdir(SASREC_BASE):
        for d in sorted(os.listdir(SASREC_BASE)):
            if "latent32" not in d or SASREC_SKIP in d:
                continue
            try:
                analyse(os.path.join(SASREC_BASE, d), f"sasrec_{d}", stride=STRIDE)
                produced.append(f"sasrec_{d}")
            except Exception as e:
                print(f"[FAIL] {d}: {type(e).__name__}: {e}", flush=True)
    else:
        print(f"[skip] no directory {SASREC_BASE}", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"produced": produced, "out_dir": OUT}, f, indent=2)
    print(f"\n=== SUMMARY: {len(produced)} tables in {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
