"""Topology tables: persistent homology before and after optimisation.

Coverage is not uniform, and the tables are split along that fact rather than
averaged across it.

H0 persistence and component counts are computed for every run - 107 of 107 -
on both the geodesic and the latent distance matrix. That is table 1.

H1 is NOT. `persistent_homology()` refuses when the estimated triangle count
exceeds max_simplices=300000 and writes -1, which means "skipped", not "zero
cycles". Measured coverage:

    scale        geodesic H1                latent H1
    300x800      36/36 runs                 36/36 runs
    1000x1200    21/36 runs                 4/36 runs
    3000x2500    0/35 runs                  0/35 runs

So table 2 reports H1 at 300x800 only, where every run has it, and names the
1000x1200 coverage instead of quoting a mean over whichever runs happened to
fit under the cap. At 3000x2500 there is nothing to report.

Table 3 is the separate subsampled H1 from `measures/h1_large_scales.json`
(900 items, 3 draws) which exists for two GI-NCF Amazon Beauty runs at the
large scales. Those numbers are comparable across epochs within a run, not
across scales, because the subsample size is fixed while the catalogue grows.
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

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = _paths.DIAGNOSTICS_DIR
MEAS = _paths.MEASURES_DIR
OUT_MD = os.path.join(_paths.TABLES_DIR, "topology_tables.md")
PAPER = _paths.FIGURES_OUT_DIR

SCALES = ["300x800", "1000x1200", "3000x2500"]
FAM_ORDER = ["GI-NCF (paper cfg)",
             "GI-SASRec (raw init)", "GI-SASRec (IDF init)"]
DS_ORDER = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery",
            "Office Products"]


def family(name):
    if name.startswith("geometry_diagnostics_ncfpaper_"):
        return "GI-NCF (paper cfg)"
    if name.startswith("geometry_diagnostics_ncf_"):
        # An lr_isomap=0.05 hyperparameter-search tree, not a paper arm:
        # deliberately excluded from every aggregate.
        return None
    if "_idf_" in name:
        return "GI-SASRec (IDF init)"
    if "lat32" in name or "latent32" in name:
        return "GI-SASRec (raw init)"
    return None


def dataset(name):
    for key, lab in (("ml1m", "MovieLens-1M"), ("ml-1m", "MovieLens-1M"),
                     ("beauty", "Amazon Beauty"), ("arts", "Arts & Crafts"),
                     ("grocery", "Grocery"), ("office", "Office Products")):
        if key in name:
            return lab
    return None


def collect():
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(DIAG, "*.csv"))):
        name = os.path.basename(f)[:-4]
        fam, ds = family(name), dataset(name)
        scale = next((s for s in SCALES if s in name), None)
        if not (fam and ds and scale) or "hidden32" in name:
            continue
        run = re.sub(r"^geometry_diagnostics_(idf|lat32|ncf|ncfpaper|extcat|sasrec)_",
                     "", name)
        if run in seen:
            continue
        seen.add(run)
        df = pd.read_csv(f).sort_values("epoch")
        rec = dict(family=fam, dataset=ds, scale=scale, run=run)
        for c in ("geo_h0_max_persistence", "lat_h0_max_persistence",
                  "geo_h1_max_persistence", "geo_h1_count",
                  "lat_h1_max_persistence", "lat_h1_count",
                  "geo_h0_count", "lat_h0_count"):
            if c not in df.columns:
                continue
            v = df[c].replace(-1, np.nan).dropna()
            if len(v) >= 2:
                rec[f"{c}_b"], rec[f"{c}_a"] = v.iloc[0], v.iloc[-1]
        rows.append(rec)
    return pd.DataFrame(rows)


def cell(g, col):
    b, a = f"{col}_b", f"{col}_a"
    if b not in g.columns or g[b].isna().all():
        return "—"
    return f"{g[b].mean():.3f} → {g[a].mean():.3f}"


def main():
    df = collect()
    md = ["# Topology of the learned manifold, before vs after optimisation", "",
          "`before` = epoch 0 (the initial approximation), `after` = the final "
          "analysed snapshot; cells are `before → after`, averaged over the "
          "available seeds. H0 persistence is the lifetime of the longest "
          "connected-component feature: larger means the cloud separates into "
          "parts that stay apart over a wider range of scales.", ""]

    # ---- table 1: H0, complete coverage
    md += ["## H0 persistence — all runs", "",
           "| dataset | model | scale | seeds | geodesic H0 | latent H0 |",
           "|---|---|---|---|---|---|"]
    for ds in DS_ORDER:
        for fam in FAM_ORDER:
            for sc in SCALES:
                g = df[(df.dataset == ds) & (df.family == fam) & (df.scale == sc)]
                if g.empty:
                    continue
                md.append(f"| {ds} | {fam} | {sc} | {len(g)} | "
                          f"{cell(g, 'geo_h0_max_persistence')} | "
                          f"{cell(g, 'lat_h0_max_persistence')} |")
    md.append("")

    # ---- table 2: H1 where it exists
    cov = []
    for sc in SCALES:
        g = df[df.scale == sc]
        cov.append((sc, len(g),
                    int(g["geo_h1_count_b"].notna().sum()) if "geo_h1_count_b" in g else 0,
                    int(g["lat_h1_count_b"].notna().sum()) if "lat_h1_count_b" in g else 0))
    md += ["## H1 — coverage first", "",
           "`h1_count = -1` in the raw CSVs means the computation was skipped "
           "for cost, not that there are no cycles. Runs with a real H1:", "",
           "| scale | runs | geodesic H1 | latent H1 |", "|---|---|---|---|"]
    for sc, n, gh, lh in cov:
        md.append(f"| {sc} | {n} | {gh}/{n} | {lh}/{n} |")
    md += ["",
           "Only 300×800 is complete, so only it is tabulated below. At "
           "3000×2500 no run has H1 at all.", "",
           "### H1 at 300×800", "",
           "| dataset | model | seeds | geodesic H1 count | geodesic H1 pers. "
           "| latent H1 count | latent H1 pers. |",
           "|---|---|---|---|---|---|---|"]
    small = df[df.scale == "300x800"]
    for ds in DS_ORDER:
        for fam in FAM_ORDER:
            g = small[(small.dataset == ds) & (small.family == fam)]
            if g.empty:
                continue
            md.append(f"| {ds} | {fam} | {len(g)} | "
                      f"{cell(g, 'geo_h1_count')} | {cell(g, 'geo_h1_max_persistence')} | "
                      f"{cell(g, 'lat_h1_count')} | {cell(g, 'lat_h1_max_persistence')} |")
    md.append("")

    # ---- table 3: subsampled H1 at large scales
    p = os.path.join(MEAS, "h1_large_scales.json")
    if os.path.exists(p):
        with open(p) as f:
            recs = json.load(f)
        md += ["## H1 at large scales — subsampled separately", "",
               "900 items drawn 3 times per snapshot, because the full "
               "computation is refused at these sizes. Mean ± std over the "
               "three draws. Comparable across epochs within a run, not across "
               "scales: the subsample is fixed while the catalogue grows.", "",
               "| run | space | epoch | H1 count | H1 max persistence |",
               "|---|---|---|---|---|"]
        for r in sorted(recs, key=lambda r: (r["run"], r["space"], r["epoch"])):
            md.append(f"| {r['run'].replace('_', ' ')} | "
                      f"{r['space'].replace('D_', '')} | {r['epoch']} "
                      f"({r['which']}) | "
                      f"{r['h1_count_mean']:.0f} ± {r['h1_count_std']:.0f} | "
                      f"{r['h1_maxpers_mean']:.4f} ± {r['h1_maxpers_std']:.4f} |")
        md.append("")

    text = "\n".join(md)
    for path in (OUT_MD, os.path.join(_paths.TABLES_DIR, "topology_tables.md")):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[Save] {os.path.normpath(path)}")
        except OSError as e:
            print(f"[not written] {path}: {e}")

    df.to_csv(os.path.join(_paths.TABLES_DIR, "topology_before_after.csv"), index=False)
    print(f"[Save] topology_before_after.csv  ({len(df)} runs)")
    print("\n" + text[:3000])


if __name__ == "__main__":
    main()
