"""Computes geometry diagnostics for every finished run that lacks a CSV.

Scope: the adopted configurations only - `lat32` and `idf`. The latent-64 runs
(`sasrec_<ds>_<scale>_seed*` with no suffix) are deliberately excluded: 32 is
the adopted configuration, 64 is history, and those 27 runs would roughly
triple the cost for results that do not enter the paper. Set INCLUDE_LAT64 to
True to add them.

Walks all three roots of the external run tree (sasrec_main, gisasrec/runs,
gisasrec/runs2), skips runs that already have a CSV, and skips runs that are
still being written - a run counts as finished only if it has >= 25 snapshots
AND a saved final model. Reading a folder mid-write is what produced two
unusable tables earlier in this project.

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
import json
import os
import sys
import time

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.ANALYSIS_DIR)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import analyze_run  # noqa: E402

S = os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external")
ROOTS = [("main", os.path.join(S, "sasrec_main")),
         ("runs", os.path.join(S, "gisasrec", "runs")),
         ("runs2", os.path.join(S, "gisasrec", "runs2"))]
OUT = _paths.DIAGNOSTICS_DIR
INCLUDE_LAT64 = False
MIN_SNAPSHOTS = 25
N_WORKERS = 6


def kind(run):
    if "_idf_" in run:
        return "idf"
    if "lat32" in run or "latent32" in run:
        return "lat32"
    return "lat64"


def main():
    os.makedirs(OUT, exist_ok=True)
    existing = {os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*.csv"))}

    todo, skipped, excluded = [], [], []
    for where, root in ROOTS:
        base = os.path.join(root, "logs_sasrec_isomap")
        if not os.path.isdir(base):
            continue
        for run in sorted(os.listdir(base)):
            p = os.path.join(base, run)
            if not os.path.isdir(p) or "smoketest" in run:
                continue
            n = len(glob.glob(os.path.join(p, "matrices_epoch*.npz")))
            fin = len(glob.glob(os.path.join(p, "*.pt")))
            if n == 0 or any(run in e for e in existing):
                continue
            k = kind(run)
            if k == "lat64" and not INCLUDE_LAT64:
                excluded.append(run)
            elif n >= MIN_SNAPSHOTS and fin > 0:
                todo.append((where, p, run, n, k))
            else:
                skipped.append((run, n, fin))

    print(f"to compute: {len(todo)}   unfinished: {len(skipped)}   "
          f"latent-64 excluded: {len(excluded)}", flush=True)
    for r, n, f in skipped:
        print(f"   SKIP (not finished) {r}: {n} snapshots, {f} models", flush=True)

    done, failed = [], []
    for i, (where, path, run, n, k) in enumerate(todo, 1):
        out_csv = os.path.join(OUT, f"geometry_diagnostics_{k}_{run}.csv")
        print(f"\n[{i}/{len(todo)}] === {where}/{run} ({n} snapshots, {k}) ===", flush=True)
        t0 = time.time()
        try:
            analyze_run(path, max_epochs_to_show=None, out_csv=out_csv, n_workers=N_WORKERS)
            print(f"[ok] {run}: {time.time() - t0:.0f}s", flush=True)
            done.append(run)
        except Exception as e:
            print(f"[FAIL] {run}: {type(e).__name__}: {e}", flush=True)
            failed.append(run)

    with open(os.path.join(OUT, "manifest_all_remaining.json"), "w") as f:
        json.dump({"done": done, "failed": failed,
                   "skipped_unfinished": skipped, "excluded_lat64": excluded}, f, indent=2)
    print(f"\n=== computed {len(done)}, failed {len(failed)} ===", flush=True)


if __name__ == "__main__":
    main()
