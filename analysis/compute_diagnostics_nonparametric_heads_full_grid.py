"""Recomputes the non-parametric baselines keeping ALL diagnostics, 3 seeds.

Two gaps are closed in one pass.

* **Fourteen measures were thrown away.** `summarize_epoch` returns all
  nineteen - ORC mean/median/fraction-negative/edges, the four spectral
  quantities, the three isometry quantities, and eight persistent-homology
  quantities. The earlier np scripts called it in full, paid the full cost,
  and wrote five columns. Everything it returns is stored here.
* **Amazon had one seed per cell.** ml-1m already has three (in
  `diag_np_baselines_seeds_results.csv`); Amazon had only seed 0. Both
  datasets get all three seeds of the winning combo.

AFTER = the snapshot nearest the best epoch recorded in
`geometry_baseline.best_epoch`, not the last snapshot. On these runs the gap
matters: validation HR peaks and then decays, so the final snapshot describes a
model nobody would select. BEFORE stays `matrices_epoch0.npz`.

Parallel over snapshots with a top-level worker, which is required for
Windows' spawn-based process creation, and a __main__ guard for the same
reason. Six workers: at 3000x2500 each holds a 2500x2500 distance matrix plus
shortest-path state.
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
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

MANUL = _paths.REPO_ROOT
MY_VER = _paths.ANALYSIS_DIR
sys.path.insert(0, MY_VER)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import summarize_epoch  # noqa: E402

OUT_DIR = _paths.TABLES_DIR
OUT_CSV = os.path.join(_paths.TABLES_DIR, "diag_np_full_results.csv")
SEEDS_ML1M = os.path.join(_paths.TABLES_DIR, "diag_np_baselines_seeds_results.csv")
INVENTORY = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")
N_WORKERS = 6


def epoch_of(p):
    return int("".join(c for c in os.path.basename(p) if c.isdigit()))


def pick_snapshots(run_dir):
    """(before_path, after_path, best_epoch, after_epoch)."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "matrices_epoch*.npz")),
                   key=epoch_of)
    if not snaps:
        return None
    best = None
    lg = os.path.join(run_dir, "run_log.json")
    if os.path.exists(lg):
        try:
            with open(lg) as f:
                best = json.load(f)["geometry_baseline"].get("best_epoch")
        except Exception:
            best = None
    if best is None:
        after = snaps[-1]
    else:
        after = min(snaps, key=lambda p: abs(epoch_of(p) - best))
    return snaps[0], after, best, epoch_of(after)


def work(task):
    """Top-level so Windows children can pickle it."""
    key, path = task
    t0 = time.time()
    try:
        with np.load(path) as d:
            out = summarize_epoch(d)
        out = {k: v for k, v in out.items() if not isinstance(v, (list, tuple))}
        return key, out, time.time() - t0, None
    except Exception as e:
        return key, {}, time.time() - t0, f"{type(e).__name__}: {e}"


def cells():
    """-> list of dicts describing every (dataset, head, scale, seed) run."""
    out = []
    if os.path.exists(SEEDS_ML1M):
        for _, r in pd.read_csv(SEEDS_ML1M).iterrows():
            out.append(dict(dataset="MovieLens-1M", head=r["head"],
                            scale=r["scale"], seed=int(r["seed"]),
                            run_dir=r["run_dir"]))
    if os.path.exists(INVENTORY):
        inv = pd.read_csv(INVENTORY)
        fin = inv[inv["status"] == "finished"]
        # group on combo_key, never on `combo`: the Amazon item_knn directory
        # names do not match their contents (one folder holds two different
        # knn_k/temp settings), so folder-name grouping merged two
        # configurations and then picked one of them arbitrarily
        for (head, scale), g in fin.groupby(["head", "scale"]):
            key = g.groupby("combo_key").hr.mean().idxmax()
            for _, r in g[g["combo_key"] == key].iterrows():
                out.append(dict(dataset="Amazon Beauty", head=head, scale=scale,
                                seed=int(r["seed"]), config=key,
                                run_dir=r["run_dir"]))
    return out


def main():
    runs = cells()
    print(f"runs: {len(runs)}", flush=True)
    by = {}
    for r in runs:
        by.setdefault((r["dataset"], r["head"], r["scale"]), []).append(r["seed"])
    for k, v in sorted(by.items()):
        print(f"   {k[0]:14s} {k[1]:9s} {k[2]:10s} seeds {sorted(v)}", flush=True)

    tasks, meta = [], {}
    for i, r in enumerate(runs):
        got = pick_snapshots(r["run_dir"])
        if got is None:
            print(f"[no snapshots] {r['run_dir']}", flush=True)
            continue
        before, after, best, after_ep = got
        meta[i] = dict(r, best_epoch=best, after_epoch=after_ep)
        tasks.append(((i, "before"), before))
        tasks.append(((i, "after"), after))

    print(f"\nsnapshots to compute: {len(tasks)}, workers {N_WORKERS}\n", flush=True)
    results, t0, done = {}, time.time(), 0
    with mp.Pool(N_WORKERS) as pool:
        for key, out, dt, err in pool.imap_unordered(work, tasks):
            done += 1
            i, side = key
            m = meta[i]
            tag = f"{m['dataset']} {m['head']} {m['scale']} seed{m['seed']} {side}"
            if err:
                print(f"[{done}/{len(tasks)}] FAIL {tag}: {err}", flush=True)
            else:
                results[key] = out
                print(f"[{done}/{len(tasks)}] ok {tag} ({dt:.0f}s, "
                      f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)

            rows = []
            for j, mm in meta.items():
                b, a = results.get((j, "before")), results.get((j, "after"))
                if b is None and a is None:
                    continue
                row = dict(mm)
                for k in sorted(set(list(b or {}) + list(a or {}))):
                    if b and k in b:
                        row[f"{k}_before"] = b[k]
                    if a and k in a:
                        row[f"{k}_after"] = a[k]
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\n[Save] {OUT_CSV}", flush=True)
    print(f"=== done in {(time.time() - t0) / 60:.0f} min ===", flush=True)


if __name__ == "__main__":
    main()
