"""Before/after summary of EVERY computed diagnostic, grouped by measure family.

Supersedes the earlier version of this script, which tabulated six measures out
of the nineteen actually in the CSVs. The missing thirteen were orc_median,
orc_f_neg, orc_n_edges, spectral_lambda_max, spectral_spectral_gap,
spectral_n_components, residual_variance and the four H0/H1 count columns.
`epoch` is an index and `orc_edges` is a raw edge list, so neither is a measure.

"Before" is the first diagnostic point of a run (epoch 0, the initial
approximation); "after" is the last.

Model families, all of which are kept separate because they are not
interchangeable:

    GI-NCF (paper cfg)    GINCF/patience_300 - lr_isomap=0.001,
                          freeze_item_projection=True, 300 outer epochs
    GI-SASRec (raw init)  rating-profile initialisation
    GI-SASRec (IDF init)  IDF-weighted initialisation
    bary-cos / ItemKNN    non-parametric heads; only five measures were
                          computed for these, the rest show as "—"

Sentinels: h1_count and h0_count use -1 for "skipped for safety", which is not
zero. Those are dropped before averaging, and each cell reports how many runs
actually contributed, so a thin cell cannot pass for a complete one.

Outputs a long-format CSV with every cell and a Markdown file with one table
per measure group.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _paths.TABLES_DIR
DIAG = _paths.DIAGNOSTICS_DIR
PAPER = _paths.FIGURES_OUT_DIR

GROUPS = [
    ("Ollivier-Ricci curvature", [
        ("orc_mean", "mean"), ("orc_median", "median"),
        ("orc_f_neg", "fraction negative"), ("orc_n_edges", "graph edges")]),
    ("Graph Laplacian spectrum", [
        ("spectral_lambda2", "lambda2"), ("spectral_lambda_max", "lambda max"),
        ("spectral_spectral_gap", "spectral gap"),
        ("spectral_n_components", "components")]),
    ("Isometry of the embedding", [
        ("kruskal_stress", "Kruskal stress"), ("spearman_rho", "Spearman rho"),
        ("residual_variance", "residual variance")]),
    ("Persistent homology", [
        ("geo_h0_max_persistence", "H0 geo pers."), ("geo_h0_count", "H0 geo count"),
        ("geo_h1_max_persistence", "H1 geo pers."), ("geo_h1_count", "H1 geo count"),
        ("lat_h0_max_persistence", "H0 lat pers."), ("lat_h0_count", "H0 lat count"),
        ("lat_h1_max_persistence", "H1 lat pers."), ("lat_h1_count", "H1 lat count")]),
]
ALL = [(c, lab) for _g, items in GROUPS for c, lab in items]
SENTINEL = {"geo_h0_count", "geo_h1_count", "lat_h0_count", "lat_h1_count"}

SCALES = ["300x800", "1000x1200", "3000x2500"]
FAM_ORDER = ["GI-NCF (paper cfg)",
             "GI-SASRec (raw init)", "GI-SASRec (IDF init)",
             "bary-cos (np)", "ItemKNN (np)"]
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


def collect_runs():
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
        rec = dict(family=fam, dataset=ds, scale=scale, run=run,
                   n_points=len(df), last_epoch=int(df.epoch.iloc[-1]))
        for c, _lab in ALL:
            if c not in df.columns:
                continue
            v = df[c]
            if c in SENTINEL:
                v = v.replace(-1, np.nan)
            v = v.dropna()
            if len(v) >= 2:
                rec[f"{c}_b"], rec[f"{c}_a"] = float(v.iloc[0]), float(v.iloc[-1])
        rows.append(rec)
    return pd.DataFrame(rows)


def collect_np():
    """The two non-parametric heads; five measures only, before/after stored."""
    rows = []
    for path, ds in ((os.path.join(ROOT, "diag_np_baselines_results.csv"),
                      "MovieLens-1M"),
                     (os.path.join(ROOT, "diag_np_amazon_results.csv"),
                      "Amazon Beauty")):
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            fam = "bary-cos (np)" if r["head"] == "bary_cos" else "ItemKNN (np)"
            rec = dict(family=fam, dataset=ds, scale=r["scale"],
                       run=str(r.get("run_dir", "")).split("\\")[-1],
                       n_points=2, last_epoch=-1)
            for c, _lab in ALL:
                for side, key in (("b", "before"), ("a", "after")):
                    col = f"{c}_{key}"
                    if col in d.columns and pd.notna(r[col]):
                        rec[f"{c}_{side}"] = float(r[col])
            rows.append(rec)
    return pd.DataFrame(rows)


def fmt(x):
    if x != x:
        return "—"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def main():
    runs = pd.concat([collect_runs(), collect_np()], ignore_index=True)
    print(f"runs included: {len(runs)}")
    print(runs.groupby(["family"]).size().to_string(), "\n")

    long = []
    for (ds, fam, sc), g in runs.groupby(["dataset", "family", "scale"]):
        for c, lab in ALL:
            b, a = f"{c}_b", f"{c}_a"
            if b not in g.columns or g[b].isna().all():
                continue
            long.append(dict(dataset=ds, family=fam, scale=sc, measure=lab,
                             column=c, n_runs=len(g),
                             n_with_value=int(g[b].notna().sum()),
                             before_mean=g[b].mean(), before_std=g[b].std(ddof=0),
                             after_mean=g[a].mean(), after_std=g[a].std(ddof=0),
                             delta=g[a].mean() - g[b].mean()))
    long = pd.DataFrame(long)
    out_csv = os.path.join(ROOT, "summary_before_after.csv")
    long.to_csv(out_csv, index=False)
    print(f"[Save] {os.path.normpath(out_csv)}  ({len(long)} rows)")

    md = ["# Every computed diagnostic, before vs after optimisation", "",
          "`before` = epoch 0 (the initial approximation), `after` = the final "
          "analysed snapshot. Cells are `before → after`, averaged over the "
          "runs that have the measure. A cell marked `(k/n)` means only k of "
          "the n runs in that row produced a value — the persistent-homology "
          "columns write a sentinel instead of a number when the complex is "
          "too large, and that sentinel is excluded rather than averaged.", "",
          "GI-NCF is `GINCF/patience_300`: lr_isomap=0.001, frozen "
          "item projection, 300 outer epochs. An earlier lr_isomap=0.05 "
          "hyperparameter-search tree exists on disk and is deliberately "
          "and is kept only so the effect of the outer learning rate is "
          "visible. The non-parametric heads had five measures computed, not "
          "nineteen.", ""]

    for gname, items in GROUPS:
        md += [f"## {gname}", "",
               "| dataset | model | scale | runs | "
               + " | ".join(lab for _c, lab in items) + " |",
               "|---|---|---|---|" + "---|" * len(items)]
        for ds in DS_ORDER:
            for fam in FAM_ORDER:
                for sc in SCALES:
                    g = runs[(runs.dataset == ds) & (runs.family == fam)
                             & (runs.scale == sc)]
                    if g.empty:
                        continue
                    cells = []
                    for c, _lab in items:
                        b, a = f"{c}_b", f"{c}_a"
                        if b not in g.columns or g[b].isna().all():
                            cells.append("—")
                            continue
                        k = int(g[b].notna().sum())
                        tag = "" if k == len(g) else f" ({k}/{len(g)})"
                        cells.append(f"{fmt(g[b].mean())} → {fmt(g[a].mean())}{tag}")
                    md.append(f"| {ds} | {fam} | {sc} | {len(g)} | "
                              + " | ".join(cells) + " |")
        md.append("")

    text = "\n".join(md)
    for path in (os.path.join(ROOT, "summary_before_after.md"),
                 os.path.join(_paths.TABLES_DIR, "summary_before_after.md")):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[Save] {os.path.normpath(path)}")
        except OSError as e:
            print(f"[not written] {path}: {e}")

    print("\ncoverage by measure (in how many cells a value exists at all):")
    tot = long.groupby("measure").n_with_value.sum()
    cells = long.groupby("measure").n_runs.sum()
    for m in [lab for _c, lab in ALL]:
        if m in tot.index:
            print(f"   {m:22s} {tot[m]:4d} of {cells[m]:4d} runs")
        else:
            print(f"   {m:22s} nowhere")


if __name__ == "__main__":
    main()
