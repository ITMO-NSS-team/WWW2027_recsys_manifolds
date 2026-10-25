"""Geometry diagnostics for the GI-NCF runs in the paper configuration.

`GINCF/patience_300` is the paper configuration: lr_isomap=0.001,
freeze_item_projection=True, 300 outer epochs, cf_epochs=200, select_by=hr.
This is the only GI-NCF tree the paper uses.

Other GI-NCF run trees may exist alongside it in EXTERNAL_RUNS_DIR; they are
hyperparameter searches and are not analysed anywhere in this repository. The
aggregation scripts select on the CSV name prefix, so only the runs produced
here are ever picked up.

Outputs are named `geometry_diagnostics_ncfpaper_*` so they sit beside the old
`ncf_*` files rather than overwriting them; both are wanted, since the pair is
what shows the learning rate's effect.

STRIDE. These runs save 31 snapshots at epochs 0,10,...,290,299. The SASRec
CSVs carry 16 points at epochs 0,20,...,280,299. Taking every second snapshot
plus the last reproduces exactly that grid. This matters for the path-length
measure, which sums |diff| between consecutive rows: 31 points and 16 points
over the same trajectory do not give comparable path lengths.

WORKERS. Four, not twelve. An Amazon np diagnostics job and eight trainings are
running; at 3000x2500 each worker holds a 2500x2500 distance matrix plus the
shortest-path state, and free RAM was ~10 GB when this was launched.

ORDER. Cheapest scale first. analyze_run writes its CSV after every completed
snapshot, so if this has to be killed the finished scales are already on disk.

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
import shutil
import sys
import tempfile
import time

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.ANALYSIS_DIR)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import analyze_run  # noqa: E402

SRC = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GINCF", "patience_300")
OUT = _paths.DIAGNOSTICS_DIR
N_WORKERS = 4
STRIDE = 2
SCALE_ORDER = ["300_800", "1000_1200", "3000_2500"]


def epoch_of(p):
    return int("".join(c for c in os.path.basename(p) if c.isdigit()))


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for scale in SCALE_ORDER:
        for seed in (0, 1, 2):
            d = os.path.join(SRC, f"{scale}_seed_{seed}")
            if os.path.isdir(d):
                runs.append((scale, seed, d))
            else:
                print(f"[no directory] {d}", flush=True)

    print(f"runs to compute: {len(runs)}  (workers {N_WORKERS}, stride {STRIDE})",
          flush=True)

    done, failed = [], []
    for i, (scale, seed, path) in enumerate(runs, 1):
        tag = f"ncf_paper_ml-1m_{scale.replace('_', 'x')}_seed{seed}"
        out_csv = os.path.join(OUT, f"geometry_diagnostics_ncfpaper_{tag}.csv")
        if os.path.exists(out_csv):
            print(f"[{i}/{len(runs)}] already present: {tag}", flush=True)
            done.append(tag)
            continue

        snaps = sorted(glob.glob(os.path.join(path, "matrices_epoch*.npz")),
                       key=epoch_of)
        if not snaps:
            print(f"[{i}/{len(runs)}] SKIP {tag}: no snapshots", flush=True)
            failed.append(tag)
            continue
        keep = snaps[::STRIDE]
        if snaps[-1] not in keep:
            keep.append(snaps[-1])

        tmp = tempfile.mkdtemp(prefix=f"diag_{tag}_")
        try:
            for s in keep:
                shutil.copy2(s, os.path.join(tmp, os.path.basename(s)))
            init = os.path.join(path, "D_input_init.npy")
            if os.path.exists(init):
                shutil.copy2(init, os.path.join(tmp, "D_input_init.npy"))
            print(f"\n[{i}/{len(runs)}] === {tag}: {len(keep)} of {len(snaps)} "
                  f"snapshots, epochs {epoch_of(keep[0])}..{epoch_of(keep[-1])} ===",
                  flush=True)
            t0 = time.time()
            analyze_run(tmp, max_epochs_to_show=None, out_csv=out_csv,
                        n_workers=N_WORKERS)
            print(f"[ok] {tag}: {time.time() - t0:.0f}s", flush=True)
            done.append(tag)
        except Exception as e:
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}", flush=True)
            failed.append(tag)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    with open(os.path.join(OUT, "manifest_ncf_paper.json"), "w") as f:
        json.dump({"source": SRC, "stride": STRIDE, "workers": N_WORKERS,
                   "done": done, "failed": failed}, f, indent=2)
    print(f"\n=== computed {len(done)}, failed {len(failed)} ===", flush=True)


if __name__ == "__main__":
    main()
