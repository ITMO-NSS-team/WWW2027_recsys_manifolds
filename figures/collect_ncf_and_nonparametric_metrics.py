"""Parses the NCF / non-parametric metrics sheet into long format and summarises it.

The sheet is laid out as one block per (dataset, seed): a header row naming
three model families - NCF, ItemKNN, Rocchio-style - each with three geometry
arms, plus a popularity column. This flattens it to
(dataset, scale, seed, family, geometry, hr, ndcg).

One quirk worth preserving rather than smoothing over: the Euclidean columns of
ItemKNN and Rocchio are identical across all three seeds. Those heads have no
trainable parameters and no random initialisation, so on a fixed split they are
deterministic - their "three seeds" are one number repeated. Standard
deviations are therefore reported as zero for them, which is correct and not a
bug, but it means a seed-spread comparison against NCF is not symmetric.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_paths.TABLES_DIR, "ncf_np_metrics_raw.csv")
OUT = os.path.join(_paths.TABLES_DIR, "ncf_np_metrics_long.csv")

FAMILIES = [("NCF", ["NCF Euclidean", "NCF Hyperbolic", "NCF Optimized (ours)"]),
            ("ItemKNN", ["Euclidean (classic)", "Hyperbolic (geoopt)",
                         "Optimized manifold (ours)"]),
            ("Rocchio", ["Euclidean (classic)", "Hyperbolic (geoopt)",
                         "Optimized manifold (ours)"])]
GEOM = ["Euclidean", "Hyperbolic", "Learned (ours)"]
SCALES = {"300/800": "300x800", "1000/1200": "1000x1200", "3000/2500": "3000x2500"}


def parse():
    rows = []
    dataset = seed = None
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            head = parts[0]
            m = re.match(r"(.+?)\s*\(seed (\d+)\)", head)
            if m:
                dataset, seed = m.group(1).strip(), int(m.group(2))
                continue
            if head not in SCALES:
                continue
            scale = SCALES[head]
            # column order: size, popularity, then 3 families x 3 geometries
            pop = parts[1]
            rows.append(dict(dataset=dataset, scale=scale, seed=seed,
                             family="Popularity", geometry="—",
                             **split_metric(pop)))
            for fi, (fam, _cols) in enumerate(FAMILIES):
                for gi, geom in enumerate(GEOM):
                    cell = parts[2 + fi * 3 + gi]
                    if not cell:
                        continue
                    rows.append(dict(dataset=dataset, scale=scale, seed=seed,
                                     family=fam, geometry=geom,
                                     **split_metric(cell)))
    return pd.DataFrame(rows)


def split_metric(cell):
    cell = re.sub(r"\(.*?\)", "", cell).strip()
    hr, ndcg = cell.split("/")
    return dict(hr=float(hr), ndcg=float(ndcg))


def main():
    df = parse()
    df.to_csv(OUT, index=False)
    print(f"[Save] {OUT}  ({len(df)} rows)")
    print("datasets:", sorted(df.dataset.unique()))
    print("families:", sorted(df.family.unique()))
    print("seeds per cell:", sorted(df.groupby(
        ["dataset", "scale", "family", "geometry"]).size().unique()))

    print("\n" + "=" * 78)
    print("MEAN OVER SEEDS (HR@10 / NDCG@10)")
    print("=" * 78)
    for ds in sorted(df.dataset.unique()):
        print(f"\n{ds}")
        for sc in ["300x800", "1000x1200", "3000x2500"]:
            pop = df[(df.dataset == ds) & (df.scale == sc) &
                     (df.family == "Popularity")].hr.mean()
            parts = []
            for fam, _ in FAMILIES:
                for geom in GEOM:
                    g = df[(df.dataset == ds) & (df.scale == sc) &
                           (df.family == fam) & (df.geometry == geom)]
                    if g.empty:
                        continue
                    parts.append(f"{fam[:4]}/{geom[:4]}={g.hr.mean():.3f}")
            print(f"  {sc:10s} pop={pop:.3f}  " + "  ".join(parts))

    print("\n" + "=" * 78)
    print("GAIN OF THE LEARNED GEOMETRY OVER EUCLIDEAN (HR@10)")
    print("=" * 78)
    for fam, _ in FAMILIES:
        print(f"\n{fam}")
        for ds in sorted(df.dataset.unique()):
            line = []
            for sc in ["300x800", "1000x1200", "3000x2500"]:
                e = df[(df.dataset == ds) & (df.scale == sc) &
                       (df.family == fam) & (df.geometry == "Euclidean")].hr
                o = df[(df.dataset == ds) & (df.scale == sc) &
                       (df.family == fam) & (df.geometry == "Learned (ours)")].hr
                h = df[(df.dataset == ds) & (df.scale == sc) &
                       (df.family == fam) & (df.geometry == "Hyperbolic")].hr
                if e.empty or o.empty:
                    continue
                line.append(f"{sc}: ours{o.mean() - e.mean():+.3f} "
                            f"hyp{h.mean() - e.mean():+.3f}")
            print(f"  {ds:16s} " + " | ".join(line))

    print("\n" + "=" * 78)
    print("SUMMARY BY FAMILY: mean margin over Euclidean, all cells")
    print("=" * 78)
    for fam, _ in FAMILIES:
        d_o, d_h, wins_o, wins_h, n = [], [], 0, 0, 0
        for ds in df.dataset.unique():
            for sc in ["300x800", "1000x1200", "3000x2500"]:
                sel = (df.dataset == ds) & (df.scale == sc) & (df.family == fam)
                e = df[sel & (df.geometry == "Euclidean")].hr.mean()
                o = df[sel & (df.geometry == "Learned (ours)")].hr.mean()
                h = df[sel & (df.geometry == "Hyperbolic")].hr.mean()
                if np.isnan(e):
                    continue
                n += 1
                d_o.append(o - e); d_h.append(h - e)
                wins_o += o > e; wins_h += h > e
        print(f"  {fam:9s} learned {np.mean(d_o):+.4f} (better than Euclidean "
              f"in {wins_o}/{n}), hyperbolic {np.mean(d_h):+.4f} "
              f"(in {wins_h}/{n})")

    print("\nare all arms above most-popular:")
    for fam, _ in FAMILIES:
        for geom in GEOM:
            bad = []
            for ds in df.dataset.unique():
                for sc in ["300x800", "1000x1200", "3000x2500"]:
                    sel = (df.dataset == ds) & (df.scale == sc)
                    v = df[sel & (df.family == fam) & (df.geometry == geom)].hr.mean()
                    p = df[sel & (df.family == "Popularity")].hr.mean()
                    if v == v and v <= p:
                        bad.append(f"{ds} {sc}")
            print(f"  {fam:9s} {geom:15s} below most-popular: "
                  f"{bad if bad else 'nowhere'}")


if __name__ == "__main__":
    main()
