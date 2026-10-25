"""Inventory of the Amazon non-parametric baseline run tree.

Read-only. Nothing is written to EXTERNAL_RUNS_DIR.

The Amazon tree differs from `GI_np_baselines_ml-1m` in three ways that matter
before any diagnostics are computed:

* runs are keyed by a numeric `n_run` instead of a `<scale>_seed_<k>` path, so
  scale and seed have to come from `run_log.json`, not from the directory name;
* several Amazon categories share one tree, so a category column is required;
* **the directory names do not match their contents.** `np_knn_k_100_np_temp_0.1`
  holds runs with knn_k=100/temp=0.1 AND knn_k=200/temp=0.05;
  `np_knn_k_20_np_temp_0.5` holds temp=0.5 and temp=0.2. Grouping by folder
  name therefore merges two distinct configurations, which is how a cell ended
  up with six runs for three seeds. The real hyperparameters are read from
  `config` and written to their own columns; `combo_key` below is what should
  be grouped on, never `combo`;
* there is no `best/` subdirectory, so "after" cannot be the best epoch the way
  it was for ml-1m. Whether a best epoch is recorded in the log at all is what
  this script checks, because comparing a *final* snapshot here against a
  *best* snapshot there would not be a like-for-like comparison.

Runs still being written are excluded by requiring `status == "finished"`:
four Arts & Crafts trainings are active while this is being written.
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
from collections import Counter

import pandas as pd

SHARE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baseline_amazon")
OUT = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")


def main():
    rows, key_union = [], Counter()
    for log in glob.glob(os.path.join(SHARE, "*", "*", "*", "run_log.json")):
        run_dir = os.path.dirname(log)
        parts = run_dir.split(os.sep)
        head, combo, n_run = parts[-3], parts[-2], parts[-1]
        try:
            with open(log) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[broken log] {run_dir}: {e}")
            continue
        cfg = d.get("config", {})
        summ = d.get("summary", {})
        key_union.update(summ.keys())

        gb = summ.get("geometry_baseline_test", {}) or {}
        rows.append(dict(
            head=head, combo=combo, n_run=n_run,
            status=d.get("status"),
            category=cfg.get("amazon_category"),
            scale=f'{cfg.get("max_users")}x{cfg.get("max_movies")}',
            seed=cfg.get("seed"),
            hr=gb.get("hr"), ndcg=gb.get("ndcg"),
            best_epoch=(d.get("geometry_baseline", {}) or {}).get("best_epoch"),
            np_head=cfg.get("np_head"), np_knn_k=cfg.get("np_knn_k"),
            np_temp=cfg.get("np_temp"), n_neighbors=cfg.get("n_neighbors"),
            lr_isomap=cfg.get("lr_isomap"),
            combo_key=(f'k{cfg.get("np_knn_k")}_t{cfg.get("np_temp")}'
                       f'_nn{cfg.get("n_neighbors")}_lr{cfg.get("lr_isomap")}'),
            n_snaps=len(glob.glob(os.path.join(run_dir, "matrices_epoch*.npz"))),
            has_best_dir=os.path.isdir(os.path.join(run_dir, "best")),
            run_dir=run_dir))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"total runs: {len(df)}   -> {os.path.normpath(OUT)}\n")

    print("statuses:", dict(df.status.value_counts()))
    print("categories:", dict(df.category.value_counts()))
    print("scales:", dict(df.scale.value_counts()))
    print("snapshots per run:", dict(df.n_snaps.value_counts()))
    print("has best/ directory:", dict(df.has_best_dir.value_counts()))
    print("best_epoch present in log:", int(df.best_epoch.notna().sum()), "of", len(df))
    print("HR present in log:", int(df.hr.notna().sum()), "of", len(df))
    print("\nsummary keys seen in the logs:")
    for k, n in key_union.most_common():
        print(f"   {k}: {n}")

    fin = df[df.status == "finished"]
    print("\ncells (category x head x scale), finished runs:")
    g = fin.groupby(["category", "head", "scale"]).agg(
        combos=("combo", "nunique"), runs=("n_run", "count"),
        hr_mean=("hr", "mean"))
    print(g.to_string())


if __name__ == "__main__":
    main()
