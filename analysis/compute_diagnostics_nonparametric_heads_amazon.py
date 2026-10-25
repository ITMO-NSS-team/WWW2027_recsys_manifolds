"""Before/after geometry diagnostics for the Amazon non-parametric baselines.

Adapted from `analysis/compute_diagnostics_nonparametric_heads_ml1m.py` (the ml-1m version),
reusing its `summarize_epoch` call and its cell-selection rule: within each
(head, scale) the winning hyperparameter combo is the one with the highest
mean test HR@10 over its three seeds, and seed 0 of that combo is the run
analysed. EXTERNAL_RUNS_DIR is read-only; nothing is written there.

Three differences from the ml-1m tree, all forced by how this one is laid out:

* **Runs are keyed by a numeric `n_run`,** not by `<scale>_seed_<k>`, so scale
  and seed are read from `run_log.json` rather than parsed from the path.
* **Only finished runs are considered** (`status == "finished"`). Four Arts &
  Crafts trainings were active while this was written; they will land in this
  same tree and must not be read mid-write.
* **"After" is the FINAL snapshot, not the best epoch.** The ml-1m runs carry a
  `best/matrices_epoch<N>.npz` and the ml-1m script used it; here there is no
  `best/` directory and no best-epoch field in any of the 72 logs. So the two
  tables are not strictly like-for-like: ml-1m's "after" is the best epoch,
  Amazon's is epoch 300. Recomputing ml-1m at its final epoch would remove the
  asymmetry and costs six extra snapshots — see RECOMPUTE_ML1M_FINAL below.

"Before" is `matrices_epoch0.npz` rather than `D_input_init.npy`, following the
ml-1m script: the bare init array carries no kNN graph or geodesic/latent pair,
so ORC, lambda2 and the persistence measures are not computable from it, while
epoch 0 is a self-consistent snapshot that differs from the init by at most one
optimizer step. The measured difference is recorded per cell so the assumption
stays checkable.
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

import numpy as np
import pandas as pd

MANUL = _paths.REPO_ROOT
MY_VER = _paths.ANALYSIS_DIR
sys.path.insert(0, MY_VER)
sys.path.insert(0, _paths.NCF_DIR)

from geometry_topology_diagnostics import summarize_epoch, _progress  # noqa: E402

SHARE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baseline_amazon")
OUT_DIR = _paths.TABLES_DIR
OUT_CSV = os.path.join(_paths.TABLES_DIR, "diag_np_amazon_results.csv")
INVENTORY = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")

MEASURES = ["orc_mean", "spectral_lambda2", "spearman_rho",
            "geo_h0_max_persistence", "lat_h0_max_persistence"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
RECOMPUTE_ML1M_FINAL = False     # see the module docstring


def epoch_of(path):
    return int("".join(c for c in os.path.basename(path) if c.isdigit()))


def pick_cells(df):
    """Winning combo per (category, head, scale), by mean HR over its seeds."""
    cells = []
    fin = df[df.status == "finished"]
    for (cat, head, scale), g in fin.groupby(["category", "head", "scale"]):
        by_combo = g.groupby("combo").hr.mean().sort_values(ascending=False)
        best_combo = by_combo.index[0]
        rows = g[(g.combo == best_combo) & (g.seed == 0)]
        if rows.empty:
            print(f"[skip] {cat} {head} {scale}: the winning combination "
                  f"{best_combo} has no seed 0")
            continue
        r = rows.iloc[0]
        cells.append(dict(category=cat, head=head, scale=scale,
                          combo=best_combo, run_dir=r.run_dir,
                          hr_test=r.hr, ndcg_test=r.ndcg,
                          combo_hr_mean=by_combo.iloc[0],
                          combo_runner_up=by_combo.index[1] if len(by_combo) > 1 else None,
                          combo_hr_gap=(by_combo.iloc[0] - by_combo.iloc[1]
                                        if len(by_combo) > 1 else None)))
    return cells


def analyze(cell):
    run_dir = cell["run_dir"]
    _progress(f"=== {cell['category']} {cell['head']} {cell['scale']} "
              f"({cell['combo']}) ===")
    snaps = sorted(glob.glob(os.path.join(run_dir, "matrices_epoch*.npz")),
                   key=epoch_of)
    before_path, after_path = snaps[0], snaps[-1]

    row = dict(cell)
    row["after_epoch"] = os.path.basename(after_path)
    row["n_snaps"] = len(snaps)

    init = os.path.join(run_dir, "D_input_init.npy")
    if os.path.exists(init):
        with np.load(before_path) as e0:
            row["d_input_init_vs_epoch0_maxdiff"] = float(
                np.max(np.abs(np.load(init) - e0["D_input"])))

    t0 = time.time()
    with np.load(before_path) as d:
        before = summarize_epoch(d)
    _progress(f"  before done ({time.time() - t0:.0f}s)")
    t0 = time.time()
    with np.load(after_path) as d:
        after = summarize_epoch(d)
    _progress(f"  after done ({time.time() - t0:.0f}s)")

    for m in MEASURES:
        b, a = before.get(m), after.get(m)
        row[f"{m}_before"], row[f"{m}_after"] = b, a
        row[f"{m}_delta"] = (a - b) if (b is not None and a is not None) else None
    return row


def main():
    if not os.path.exists(INVENTORY):
        print(f"no inventory {INVENTORY} - run inventory_nonparametric_amazon_runs.py first")
        return
    df = pd.read_csv(INVENTORY)
    cells = pick_cells(df)
    print(f"cells to compute: {len(cells)}\n")
    for c in cells:
        gap = c["combo_hr_gap"]
        print(f"  {c['category']} {c['head']:9s} {c['scale']:10s} "
              f"-> {c['combo']}  (HR {c['combo_hr_mean']:.4f}"
              + (f", margin over {c['combo_runner_up']}: {gap:+.4f})" if gap is not None else ")"))

    results = []
    for i, c in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}]", flush=True)
        try:
            results.append(analyze(c))
        except Exception as e:
            print(f"[FAIL] {c['head']} {c['scale']}: {type(e).__name__}: {e}",
                  flush=True)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)   # save as we go
    print(f"\n[Save] {OUT_CSV}  ({len(results)} cells)")

    if results:
        r = pd.DataFrame(results)
        print("\nsummary (before -> after):")
        for _, x in r.iterrows():
            parts = [f"{m.replace('_max_persistence', '')}: "
                     f"{x[m + '_before']:.3f}->{x[m + '_after']:.3f}"
                     for m in MEASURES if pd.notna(x.get(m + "_before"))]
            print(f"  {x['head']:9s} {x['scale']:10s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
