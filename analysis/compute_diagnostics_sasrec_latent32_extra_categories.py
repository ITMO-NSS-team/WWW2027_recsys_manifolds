"""
Geometry/topology diagnostics for the lat32 SASRec runs on
Arts & Crafts, Grocery & Gourmet Food and Office Products.

latent 32 only - that is the adopted configuration; the latent-64 runs are
history and cost more than everything else combined.

Reads snapshots from EXTERNAL_RUNS_DIR and writes the CSVs into
results/diagnostics/. EXTERNAL_RUNS_DIR is never written to.

Runs whose snapshot count falls short of the expected 31 are SKIPPED rather
than analysed partially: those are jobs still being written, and analysing a
folder mid-write is exactly the mistake that produced the two unusable
ml1m_3000x2500 tables already flagged in the README.

__main__ guard is mandatory - analyze_run forks workers and on Windows the
children re-import this module.
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

SHARE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external",
                    "sasrec_main", "logs_sasrec_isomap")
OUT = _paths.DIAGNOSTICS_DIR
MIN_SNAPSHOTS = 25      # complete runs carry 31; anything well below is still being written
N_WORKERS = 6


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = sorted(d for d in os.listdir(SHARE)
                  if "lat32" in d and "smoketest" not in d
                  and os.path.isdir(os.path.join(SHARE, d)))

    todo, skipped = [], []
    for r in runs:
        n = len(glob.glob(os.path.join(SHARE, r, "matrices_epoch*.npz")))
        (todo if n >= MIN_SNAPSHOTS else skipped).append((r, n))

    print(f"to compute: {len(todo)} runs, incomplete skipped: {len(skipped)}", flush=True)
    for r, n in skipped:
        print(f"   SKIP {r}: {n} snapshots (31 expected)", flush=True)

    done, failed = [], []
    for i, (r, n) in enumerate(todo, 1):
        out_csv = os.path.join(OUT, f"geometry_diagnostics_extcat_{r}.csv")
        if os.path.exists(out_csv):
            print(f"[{i}/{len(todo)}] already present: {r}", flush=True)
            done.append(r)
            continue
        print(f"\n[{i}/{len(todo)}] === {r} ({n} snapshots) ===", flush=True)
        t0 = time.time()
        try:
            analyze_run(os.path.join(SHARE, r), max_epochs_to_show=None,
                        out_csv=out_csv, n_workers=N_WORKERS)
            print(f"[ok] {r}: {time.time() - t0:.0f}s", flush=True)
            done.append(r)
        except Exception as e:
            print(f"[FAIL] {r}: {type(e).__name__}: {e}", flush=True)
            failed.append(r)

    with open(os.path.join(OUT, "manifest_extcat_lat32.json"), "w") as f:
        json.dump({"source": SHARE, "latent": 32, "done": done, "failed": failed,
                   "skipped_incomplete": skipped}, f, indent=2)
    print(f"\n=== SUMMARY: computed {len(done)}, failed {len(failed)}, "
          f"incomplete skipped {len(skipped)} ===", flush=True)


if __name__ == "__main__":
    main()
