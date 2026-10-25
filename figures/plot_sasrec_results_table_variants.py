"""Three candidate figures for the SASRec results table.

The table is 5 datasets x 3 scales x 5 arms x 3 seeds = 225 numbers. Printed in
full it costs a page and nobody reads it; printed as means it hides the seed
spread, which on several cells is the whole story. These three forms each trade
that off differently - pick one for the paper and keep the full table in an
appendix.

A  absolute HR@10, one panel per dataset, seeds as individual dots.
   Complete and literal; the reader sees every number's position. Costs width,
   and the Poincare arm sits so far below the rest that it compresses everyone
   else into the top third of each panel.

B  HR@10 relative to classic SASRec, one row per dataset. The baseline becomes
   a zero line, so all 15 cells share one axis and "who beats the baseline" is
   read directly. The most compact of the three, and the closest to the
   question the paper asks. Poincare is off-scale by construction and is
   reported in the caption instead of being drawn.

C  mean + 95% CI per arm pooled over datasets, one panel per scale. Smallest of
   all, but pooling across datasets is only honest because every arm sees the
   same datasets; it hides which dataset drives which effect.

Seeds are drawn as open dots with a horizontal mean bar, never as error bars:
with n=3 an error bar implies a distributional claim the data cannot support,
while three dots show exactly what was observed.
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
ARMS = [
    ("classic", "Euclidean (classic)", "#2a78d6"),
    ("poincare", "Fixed Poincaré", "#8a8a85"),
    ("hypsasrec", "HypSASRec", "#eb6834"),
    ("ours32", "Ours (raw init)", "#1baf7a"),
    ("ours32idf", "Ours (IDF init)", "#4a3aa7"),
]
DATASETS = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery", "Office Products"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
SCALE_LBL = {s: s.replace("x", "×") for s in SCALES}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


def variant_a(df):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        for si, sc in enumerate(SCALES):
            for ai, (arm, _lab, col) in enumerate(ARMS):
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)].hr.values
                if not len(v):
                    continue
                x = si + (ai - 2) * 0.16
                ax.scatter([x] * len(v), v, s=26, facecolor="none", edgecolor=col,
                           linewidth=1.4, zorder=3)
                ax.plot([x - 0.06, x + 0.06], [v.mean()] * 2, color=col, lw=2.4, zorder=4)
        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels([SCALE_LBL[s] for s in SCALES], fontsize=9)
        ax.set_title(ds, fontsize=11, color=INK, pad=8)
    axes[0].set_ylabel("HR@10", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                          markeredgecolor=c, markeredgewidth=1.5, markersize=7, label=l)
               for _a, l, c in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("SASRec quality by geometry — every seed shown",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "table_A_absolute.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variant_b(df):
    arms = [a for a in ARMS if a[0] != "poincare"]
    fig, axes = plt.subplots(5, 1, figsize=(8.4, 9.4), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        ax.axhline(0, color=INK2, lw=1.1, zorder=2)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        for si, sc in enumerate(SCALES):
            base = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "classic")].hr.mean()
            for ai, (arm, _lab, col) in enumerate(arms):
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)].hr.values - base
                if not len(v):
                    continue
                x = si + (ai - 1.5) * 0.17
                ax.scatter([x] * len(v), v, s=28, facecolor="none", edgecolor=col,
                           linewidth=1.5, zorder=3)
                ax.plot([x - 0.065, x + 0.065], [v.mean()] * 2, color=col, lw=2.6, zorder=4)
        ax.set_ylabel(ds, fontsize=9.5, color=INK, rotation=0, ha="right", va="center")
        ax.set_ylim(-0.21, 0.13)
    axes[-1].set_xticks(range(len(SCALES)))
    axes[-1].set_xticklabels([SCALE_LBL[s] for s in SCALES], fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                          markeredgecolor=c, markeredgewidth=1.5, markersize=7, label=l)
               for _a, l, c in arms]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("HR@10 relative to classic SASRec (zero line); each dot is one seed",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "table_B_relative.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variant_c(df):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    for ax, sc in zip(axes, SCALES):
        style(ax)
        ax.axvline(0, color=INK2, lw=1.1, zorder=2)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        arms = [a for a in ARMS if a[0] not in ("classic", "poincare")]
        for ai, (arm, _lab, col) in enumerate(arms):
            deltas = []
            for ds in DATASETS:
                base = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "classic")].hr.mean()
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)].hr.values
                if len(v):
                    deltas.append(v.mean() - base)
            if not deltas:
                continue
            d = np.array(deltas)
            y = len(arms) - 1 - ai
            ax.scatter(d, [y] * len(d), s=46, facecolor="none", edgecolor=col,
                       linewidth=1.5, zorder=3)
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            ax.plot([m - 1.96 * se, m + 1.96 * se], [y, y], color=col, lw=2.6, zorder=4)
            ax.scatter([m], [y], s=90, color=col, marker="|", linewidth=2.6, zorder=5)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([l for _a, l, _c in arms][::-1], fontsize=9.5)
        ax.set_title(SCALE_LBL[sc], fontsize=11, color=INK, pad=8)
        ax.set_xlabel("HR@10 − classic", fontsize=9.5, color=INK2)
    fig.suptitle("Mean gap to classic SASRec per scale; dots are datasets, bar is the 95% CI",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "table_C_pooled.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    df = pd.read_csv(CSV)
    variant_a(df); variant_b(df); variant_c(df)
    print("done:", ", ".join(f for f in sorted(os.listdir(OUTDIR)) if f.startswith("table_")))
    print("\nmean margin over classic by scale:")
    for sc in SCALES:
        parts = []
        for arm, lab, _c in ARMS:
            if arm in ("classic", "poincare"):
                continue
            ds_d = []
            for ds in DATASETS:
                base = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "classic")].hr.mean()
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)].hr.values
                if len(v):
                    ds_d.append(v.mean() - base)
            parts.append(f"{lab}: {np.mean(ds_d):+.4f}")
        print(f"  {sc:11s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
