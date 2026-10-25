"""Main results figure: SASRec quality by geometry, across datasets and scales.

Layout: 5 columns (datasets) x 2 rows (HR@10 on top, NDCG@10 below). Both
metrics share the x axis and the colour mapping, so the legend is read once and
the two rows can be compared directly. Two rows rather than two y-axes on one
plot: a dual-axis chart would make the crossing point of the two metrics an
artefact of the scaling.

Each arm at each scale is drawn as a cross - a vertical bar spanning the three
seeds with a horizontal cap at their mean - and the means are joined across
scales by a line in the same colour. The spread is therefore the observed range
of three runs, not a confidence interval: with n=3 an error bar would imply a
distributional claim the data does not support.

The popularity baseline is model-free and deterministic given the split, so it
has no spread and is drawn as a dashed grey reference rather than as a series.
It is worth having on the plot: the fixed-Poincare arm sits at or below it on
every dataset, which is the clearest statement of how little that arm's
distances carry.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = _paths.FIGURES_OUT_DIR
CSV = os.path.join(_paths.TABLES_DIR, "sasrec_metrics_table.csv")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#deddd8"
REF = "#8a8a85"
# validated categorical palette, fixed slot order
ARMS = [
    ("classic", "Euclidean (classic)", "#2a78d6"),
    ("hypsasrec", "HypSASRec", "#eb6834"),
    ("ours32", "Ours (raw init)", "#1baf7a"),
    ("ours32idf", "Ours (IDF init)", "#eda100"),
    ("poincare", "Fixed Poincaré", "#4a3aa7"),
]
DATASETS = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery", "Office Products"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
SCALE_LBL = ["300×800", "1000×1200", "3000×2500"]
METRICS = [("hr", "HR@10"), ("ndcg", "NDCG@10")]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True)
    for row, (metric, mlabel) in enumerate(METRICS):
        for col, ds in enumerate(DATASETS):
            ax = axes[row, col]
            style(ax)

            pop = [df[(df.dataset == ds) & (df.scale == sc) &
                      (df.arm == "popularity")][metric].mean() for sc in SCALES]
            ax.plot(range(3), pop, color=REF, lw=1.6, ls=(0, (4, 3)), zorder=2,
                    label="Popularity" if (row == 0 and col == 0) else None)

            for ai, (arm, lab, c) in enumerate(ARMS):
                off = (ai - 2) * 0.085
                means = []
                for si, sc in enumerate(SCALES):
                    v = df[(df.dataset == ds) & (df.scale == sc) &
                           (df.arm == arm)][metric].values
                    if not len(v):
                        means.append(np.nan)
                        continue
                    x = si + off
                    means.append(v.mean())
                    ax.plot([x, x], [v.min(), v.max()], color=c, lw=1.4, zorder=3,
                            solid_capstyle="round")
                    ax.plot([x - 0.055, x + 0.055], [v.mean()] * 2, color=c, lw=2.1,
                            zorder=4, solid_capstyle="round")
                ax.plot(np.arange(3) + off, means, color=c, lw=1.3, alpha=0.75, zorder=2,
                        label=lab if (row == 0 and col == 0) else None)

            if row == 0:
                ax.set_title(ds, fontsize=11.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=11, color=INK)
            ax.set_xticks(range(3))
            ax.set_xticklabels(SCALE_LBL, fontsize=8.5)
            ax.set_xlim(-0.45, 2.45)
        lo = min(df[df[metric].notna()][metric].min(), 0)
        hi = df[metric].max()
        for col in range(5):
            axes[row, col].set_ylim(lo - 0.02, hi + 0.03)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(l) for l in
             ["Euclidean (classic)", "HypSASRec", "Ours (raw init)",
              "Ours (IDF init)", "Fixed Poincaré", "Popularity"] if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=6, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("SASRec quality under five embedding geometries",
                 fontsize=14, color=INK, y=1.0)
    fig.text(0.5, -0.085, "Cross = range over three seeds, cap at the mean; lines join the "
             "means across scales. Popularity is deterministic given the split and has no spread.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"main_results_sasrec.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("[Save] main_results_sasrec.png / .pdf")

    print("\nPoincare against most-popular (HR@10, mean):")
    for ds in DATASETS:
        parts = []
        for sc in SCALES:
            p = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "poincare")].hr.mean()
            q = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "popularity")].hr.mean()
            parts.append(f"{sc}: {p:.3f} vs {q:.3f} {'higher' if p > q else 'LOWER'}")
        print(f"  {ds:16s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
