"""How far apart are the best epoch and the final epoch, per run?

Every before/after table in this project takes "after" to be the LAST analysed
snapshot. If a run's validation HR peaks and then decays, that choice describes
a degraded model and calls it the trained one. This measures the gap across
every family that stores a per-epoch history, so the decision to switch to the
best epoch can be made on evidence rather than per run.
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

import numpy as np

MY_VER = _paths.ANALYSIS_DIR
SASREC = _paths.SASREC_RUNS_DIR
NCF_ML1M = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GINCF", "patience_300")


def report(label, runs):
    print("=" * 92)
    print(label)
    print("=" * 92)
    print(f'{"run":52s} {"epochs":>5s} {"best":>5s} {"HR best":>8s} '
          f'{"HR last":>8s} {"loss":>8s}')
    drops = []
    for name, hr in runs:
        if hr is None or not len(hr):
            print(f"{name:52s}  no history")
            continue
        b = int(np.nanargmax(hr))
        drop = (hr[b] - hr[-1]) / hr[b] if hr[b] else np.nan
        drops.append(drop)
        flag = "  <-- collapse" if drop > 0.25 else ""
        print(f"{name:52s} {len(hr):5d} {b:5d} {hr[b]:8.4f} {hr[-1]:8.4f} "
              f"{drop:7.1%}{flag}")
    if drops:
        print(f"\n  median loss {np.median(drops):.1%}, "
              f"maximum {np.max(drops):.1%}, "
              f"runs losing >25%: {sum(d > 0.25 for d in drops)}/{len(drops)}")
    print()


def from_history_json(path, key="val_hr"):
    try:
        with open(path) as f:
            h = json.load(f)
        if isinstance(h, dict) and key in h:
            return np.asarray(h[key], dtype=float)
    except Exception:
        pass
    return None


def main():
    # GI-NCF on Amazon Beauty, paper config, histories next to the snapshots
    runs = []
    base = os.path.join(MY_VER, "logs_movielens_isomap_cf")
    for d in sorted(glob.glob(os.path.join(base, "amazon_beauty_frozenproj_*_0.001"))):
        if not os.path.exists(os.path.join(d, "geometry_diagnostics.csv")):
            continue
        runs.append((os.path.basename(d)[:52],
                     from_history_json(os.path.join(d, "history.json"))))
    report("GI-NCF, Amazon Beauty (the runs that have diagnostics)", runs)

    # GI-NCF on ml-1m, paper config
    runs = []
    for d in sorted(glob.glob(os.path.join(NCF_ML1M, "*_seed_*"))):
        lg = os.path.join(d, "run_log.json")
        hr = None
        if os.path.exists(lg):
            try:
                with open(lg) as f:
                    hr = np.asarray(json.load(f)["gi"]["history"]["val_hr"],
                                    dtype=float)
            except Exception:
                pass
        runs.append((os.path.basename(d), hr))
    report("GI-NCF, ml-1m, patience_300", runs)

    # GI-SASRec
    runs = []
    for d in sorted(glob.glob(os.path.join(SASREC, "*latent32*"))):
        runs.append((os.path.basename(d)[:52],
                     from_history_json(os.path.join(d, "history.json"))))
    report("GI-SASRec (local run tree, latent32)", runs)


if __name__ == "__main__":
    main()
