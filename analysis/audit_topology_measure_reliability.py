"""What topology we actually have, before tabulating any of it.

The persistent-homology columns carry a sentinel: `h1_count == -1` means
"skipped for safety" because the estimated triangle count exceeded
max_simplices, which happens from about 2500 items upward. Averaging over it
silently reports -1 cycles. This audits, per model family and scale, how many
runs have real H1 at all, so the table that follows can state coverage instead
of implying it.

Also checks `measures/h1_large_scales.json`, the subsampled H1 computed
separately for the scales where the in-pipeline computation bails out.
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
MEASURES_DIR = _paths.MEASURES_DIR

TOPO = ["geo_h0_max_persistence", "geo_h0_count", "geo_h1_max_persistence",
        "geo_h1_count", "lat_h0_max_persistence", "lat_h0_count",
        "lat_h1_max_persistence", "lat_h1_count"]
SCALES = ["300x800", "1000x1200", "3000x2500"]


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


def main():
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
        rec = dict(family=fam, dataset=ds, scale=scale, run=run, n=len(df))
        for c in TOPO:
            if c not in df.columns:
                continue
            v = df[c]
            rec[f"{c}_sentinel"] = int((v == -1).sum())
            vv = v.replace(-1, np.nan).dropna()
            if len(vv) >= 2:
                rec[f"{c}_before"] = vv.iloc[0]
                rec[f"{c}_after"] = vv.iloc[-1]
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"runs: {len(df)}\n")
    print("H1 COVERAGE: share of sentinel rows (-1) by scale")
    for c in ("geo_h1_count", "lat_h1_count"):
        col = f"{c}_sentinel"
        if col not in df.columns:
            continue
        print(f"\n  {c}")
        for sc in SCALES:
            g = df[df.scale == sc]
            if g.empty:
                continue
            tot = g.n.sum()
            sent = g[col].sum()
            real = g[f"{c}_before"].notna().sum() if f"{c}_before" in g else 0
            print(f"    {sc:11s} sentinels {sent}/{tot} rows, "
                  f"runs with real H1: {real}/{len(g)}")

    print("\n\nWHAT IS AVAILABLE FOR H0 (always computed):")
    for c in ("geo_h0_max_persistence", "lat_h0_max_persistence",
              "geo_h0_count", "lat_h0_count"):
        b = f"{c}_before"
        if b in df.columns:
            print(f"  {c:28s} filled in {df[b].notna().sum()}/{len(df)} runs")

    h1f = os.path.join(MEASURES_DIR, "h1_large_scales.json")
    print(f"\n\nmeasures/h1_large_scales.json: ", end="")
    if os.path.exists(h1f):
        with open(h1f) as f:
            d = json.load(f)
        print(f"present, {len(d)} top-level entries")
        print("  keys:", list(d)[:6])
        k0 = list(d)[0]
        print(f"  example {k0}: {json.dumps(d[k0])[:300]}")
    else:
        print("absent")

    df.to_csv(os.path.join(_paths.TABLES_DIR, "topology_audit.csv"), index=False)
    print(f"\n[Save] topology_audit.csv")


if __name__ == "__main__":
    main()
