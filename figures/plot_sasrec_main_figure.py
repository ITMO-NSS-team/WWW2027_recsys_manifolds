"""Main results figure, single row: HR@10 by geometry across datasets and scales.

NDCG@10 is dropped - it tracked HR@10 almost exactly in the two-row version, so
the second row cost half the figure for no additional claim. The full table in
the appendix keeps both.

Notation follows the project's established geometry plots: a black-edged marker
per arm at each scale, its own shape so the arms stay distinguishable in
greyscale and for colour-vision deficiency, with the three seeds behind it as
small translucent dots of the same colour. Means are joined across scales by a
line in the series colour.

Each panel carries its own y range. Shared limits flattened Grocery, where the
four competitive arms sit within 0.01 of each other, into a single line.

Two variants are written. The full one keeps Fixed Poincare, which sits about
four times below every other arm and therefore spends most of the vertical
space on an empty band. The `_nopoincare` one drops it, which roughly triples
the resolution of the competitive band; the arm then belongs in the caption or
in the appendix table rather than in the plot.

Popularity is model-free and deterministic given the split, so it carries no
spread and is drawn as a dashed grey reference rather than as a series.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os
import shutil
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = _paths.FIGURES_OUT_DIR
CSV = os.path.join(_paths.TABLES_DIR, "sasrec_metrics_table.csv")
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#deddd8", "#8a8a85"
# the compact variant drops the in-figure title: matplotlib reserves a band for
# suptitle on top of tight_layout's own padding, and that band was the last
# large piece of empty space in the row. The LaTeX caption carries the title.
TITLE_IN_COMPACT = False


def save(fig, fname):
    """Render to a local temp file, then copy into the (cloud-synced) paper
    folder. Writing there directly fails intermittently with EINVAL while the
    sync client holds the path, which silently cost one render already."""
    tmp = os.path.join(tempfile.gettempdir(), fname)
    fig.savefig(tmp, dpi=200, bbox_inches="tight", facecolor="white")
    dst = os.path.join(OUTDIR, fname)
    for attempt in range(5):
        try:
            shutil.copy2(tmp, dst)
            return
        except OSError as e:
            if attempt == 4:
                raise
            print(f"   [retry {attempt + 1}] {type(e).__name__}: {e}", flush=True)
            time.sleep(1.5)

# Both "ours" arms are crosses in two shades of green: they are the same method
# under two initialisations, and the shared hue says so. The two greens are not
# separated enough to carry identity on their own, so the marker SHAPE does it -
# a filled plus against a filled x - which is the documented way to use a
# marginal colour pair.
ARMS = [
    ("classic",   "SASRec classic",              "#2a78d6", "s"),
    ("hypsasrec", "HypSASRec",                   "#eb6834", "D"),
    ("ours32",    "Learned Manifold (ours)",     "#1baf7a", "P"),
    ("ours32idf", "Learned Manifold IDF (ours)", "#00702f", "X"),
    ("poincare",  "Fixed Poincaré",              "#4a3aa7", "^"),
]
DATASETS = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery", "Office Products"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
ORDER = ["SASRec classic", "HypSASRec", "Learned Manifold (ours)",
         "Learned Manifold IDF (ours)", "Fixed Poincaré", "Popularity"]


def render(df, arms, fname, refs_as_text=False, compact=False):
    """refs_as_text moves Poincare and Popularity out of the plot and into a
    line of panel text. Both are flat references rather than competing series,
    and on Grocery they sit so far below the four arms that drawing them spends
    two thirds of the panel height on an empty band.

    compact squeezes the row: per-panel y ranges mean every panel carries its
    own tick labels, and at the default spacing those labels plus the inter-axes
    gap were costing more width than the plots between them."""
    rng = np.random.default_rng(0)
    fs = dict(title=11, tick=8.5, xtick=8.5, ylab=11, leg=10, note=8.5) if compact         else dict(title=12, tick=9.5, xtick=9, ylab=12, leg=11, note=9)
    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.6) if compact else (17.5, 4.6))
    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=fs["tick"], length=3,
                       pad=1.5 if compact else 3.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

        pop = [df[(df.dataset == ds) & (df.scale == sc) & (df.arm == "popularity")].hr.mean()
               for sc in SCALES]
        if not refs_as_text:
            ax.plot(range(3), pop, color=REF, lw=1.6, ls=(0, (4, 3)), zorder=2,
                    label="Popularity" if col == 0 else None)

        for arm, lab, c, mk in arms:
            means = []
            for si, sc in enumerate(SCALES):
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)].hr.values
                if not len(v):
                    means.append(np.nan)
                    continue
                means.append(v.mean())
                ax.scatter(si + rng.uniform(-0.1, 0.1, len(v)), v, s=26, color=c,
                           alpha=0.45, linewidth=0, zorder=3)
            # the fixed-Poincare arm is an ablation, not a competitor: dashed
            ax.plot(range(3), means, color=c, lw=1.6, alpha=0.85, zorder=4,
                    ls=(0, (5, 2)) if arm == "poincare" else "-")
            ax.scatter(range(3), means, s=68, color=c, marker=mk, edgecolor=INK,
                       linewidth=0.9, zorder=5, label=lab if col == 0 else None)

        ax.set_title(ds, fontsize=fs["title"], color=INK, pad=3 if compact else 9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SCALES, fontsize=fs["xtick"])
        ax.set_xlim(-0.4, 2.4)
        keep = [a[0] for a in arms] + ([] if refs_as_text else ["popularity"])
        vals = df[(df.dataset == ds) & (df.arm.isin(keep))].hr
        lo, hi = vals.min(), vals.max()
        pad = 0.07 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="y", labelleft=True)
        if compact:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

        if refs_as_text:
            # below the panel, not inside it: inside, the block collided with the
            # low left-hand end of the Office Products and Grocery series
            pc = df[(df.dataset == ds) & (df.arm == "poincare")].hr
            ax.text(0.5, -0.195,
                    f"Fixed Poincaré {pc.min():.2f}–{pc.max():.2f}   ·   "
                    f"Popularity {min(pop):.2f}–{max(pop):.2f}",
                    transform=ax.transAxes, fontsize=8.5, color=INK2,
                    va="top", ha="center")
    axes[0].set_ylabel("test HR@10", fontsize=fs["ylab"], color=INK,
                       fontweight="bold", labelpad=2 if compact else 4)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(l) for l in ORDER if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=len(order), frameon=False, fontsize=fs["leg"],
               columnspacing=1.4 if compact else 2.0, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.045 if compact else -0.07))
    # no count in the title: the variants plot four or five arms and a fixed
    # number would contradict the legend
    if TITLE_IN_COMPACT or not compact:
        fig.suptitle("SASRec quality across embedding geometries",
                     fontsize=13 if compact else 14, color=INK,
                     y=1.0 if compact else 1.01)
    fig.text(0.5, -0.095 if compact else -0.125, "Large marker = mean over three seeds, small dots = the seeds "
             "themselves. Each panel has its own y range. " +
             ("Fixed Poincaré and Popularity are far below every arm shown and are "
              "quoted in each panel instead of plotted."
              if refs_as_text else
              "Popularity is deterministic given the split and has no spread."),
             ha="center", fontsize=fs["note"], color=INK2)
    fig.tight_layout(w_pad=0.1 if compact else 1.0)
    save(fig, fname)
    plt.close(fig)
    print("[Save]", fname)


def main():
    df = pd.read_csv(CSV)
    render(df, ARMS, "main_results_hr.png", compact=True)
    comp_arms = [a for a in ARMS if a[0] != "poincare"]
    render(df, comp_arms, "main_results_hr_nopoincare.png")
    render(df, comp_arms, "main_results_hr_stretched.png", refs_as_text=True)

    # how much of each panel's height the four competitive arms actually occupy
    comp = [a[0] for a in ARMS if a[0] != "poincare"]
    print("\nshare of panel height taken by the competitive arms:")
    for ds in DATASETS:
        s = df[df.dataset == ds]
        full = s.hr.max() - s.hr.min()
        kept = s[s.arm.isin(comp + ["popularity"])]
        band = s[s.arm.isin(comp)]
        w = band.hr.max() - band.hr.min()
        print(f"  {ds:16s} with all {w / full:5.0%}   without Poincare "
              f"{w / (kept.hr.max() - kept.hr.min()):5.0%}")

    print("\nGrocery: within-arm spread against between-arm difference (HR@10):")
    for sc in SCALES:
        s = df[(df.dataset == "Grocery") & (df.scale == sc)]
        ms = {a: s[s.arm == a].hr.mean() for a in comp}
        spread = max(s[s.arm == a].hr.max() - s[s.arm == a].hr.min() for a in comp)
        print(f"  {sc:10s} between arms {max(ms.values()) - min(ms.values()):.4f}, "
              f"within an arm up to {spread:.4f}  -> "
              f"{'no winner' if spread > max(ms.values()) - min(ms.values()) else 'real gap'}")


if __name__ == "__main__":
    main()
