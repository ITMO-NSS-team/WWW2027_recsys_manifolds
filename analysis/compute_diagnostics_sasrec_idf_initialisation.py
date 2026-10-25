"""Geometry diagnostics for the IDF-initialised GI-SASRec runs.

These are the runs that replace the raw rating-profile initialisation with
IDF-weighted item features (see shared/idf_weighted_item_features.py). The point of
measuring them is to see whether the four reliable measures - ORC, algebraic
connectivity, isometry, latent H0 persistence - move the same way under the
better-conditioned initialisation, or whether the previously measured
trajectories were partly an artefact of the NaN guard firing on every epoch.

17 of 18 runs are analysed: `sasrec_grocery_idf_3000x2500_seed2` is skipped,
it has 26 snapshots and no final model, i.e. it never finished.

Reads from EXTERNAL_RUNS_DIR, writes CSVs into this repository. __main__ guard is
mandatory: analyze_run forks workers and Windows children re-import the module.
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

BASE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external", "gisasrec")
# runs2 was added later and holds the 300x800 scale plus ml-1m and Beauty
SHARES = [os.path.join(BASE, w, "logs_sasrec_isomap") for w in ("runs", "runs2")]
OUT = _paths.DIAGNOSTICS_DIR
MIN_SNAPSHOTS = 30      # finished runs carry 30-31
N_WORKERS = 6


def main():
    os.makedirs(OUT, exist_ok=True)
    todo, skipped = [], []
    for share in SHARES:
        if not os.path.isdir(share):
            continue
        for r in sorted(d for d in os.listdir(share)
                        if "idf" in d and os.path.isdir(os.path.join(share, d))):
            n = len(glob.glob(os.path.join(share, r, "matrices_epoch*.npz")))
            fin = len(glob.glob(os.path.join(share, r, "*.pt")))
            # unfinished runs are skipped, not analysed partially - reading a folder
            # that is still being written is what produced two unusable tables before
            (todo if (n >= MIN_SNAPSHOTS and fin > 0) else skipped).append((share, r, n, fin))

    print(f"to compute: {len(todo)}, unfinished skipped: {len(skipped)}", flush=True)
    for _sh, r, n, fin in skipped:
        print(f"   SKIP {r}: {n} snapshots, {fin} final models", flush=True)

    done, failed = [], []
    for i, (share, r, n, _f) in enumerate(todo, 1):
        out_csv = os.path.join(OUT, f"geometry_diagnostics_idf_{r}.csv")
        if os.path.exists(out_csv):
            print(f"[{i}/{len(todo)}] already present: {r}", flush=True)
            done.append(r)
            continue
        print(f"\n[{i}/{len(todo)}] === {r} ({n} snapshots) ===", flush=True)
        t0 = time.time()
        try:
            analyze_run(os.path.join(share, r), max_epochs_to_show=None,
                        out_csv=out_csv, n_workers=N_WORKERS)
            print(f"[ok] {r}: {time.time() - t0:.0f}s", flush=True)
            done.append(r)
        except Exception as e:
            print(f"[FAIL] {r}: {type(e).__name__}: {e}", flush=True)
            failed.append(r)

    with open(os.path.join(OUT, "manifest_idf.json"), "w") as f:
        json.dump({"sources": SHARES, "done": done, "failed": failed,
                   "skipped_unfinished": [(r, n, f_) for _s, r, n, f_ in skipped]}, f, indent=2)
    print(f"\n=== computed {len(done)}, failed {len(failed)}, "
          f"skipped {len(skipped)} ===", flush=True)


if __name__ == "__main__":
    main()
