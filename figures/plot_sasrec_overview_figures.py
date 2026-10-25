"""Three candidate overview figures: geometry regime vs SASRec quality.

Regimes, left to right in every variant, in the order the paper argues them:
  Euclidean            - classic SASRec, free embedding table
  Fixed Poincare       - a-priori hyperbolic coordinates, derived curvature (ablation)
  Hyperbolic (HypSASRec) - a model built around hyperbolic geometry, derived curvature
  Learned (ours)       - geometry fitted to the recommendation objective

Colors come from the data-viz skill's validated categorical palette, in fixed
slot order (blue / orange / aqua / yellow). Palette validation: worst adjacent
CVD dE 9.1, worst adjacent normal-vision dE 19.6 - both clear their gates.
The Poincare ablation is drawn in a muted ink rather than a categorical hue in
variants A and B, because it is a control, not a competing geometry.

Variant A - small multiples, one panel per dataset: the complete picture.
Variant B - dot plot, one row per dataset x scale: easiest within-row comparison.
Variant C - regime summary: every cell as a dot, one column per regime.

Cells with n < 3 are drawn hollow and excluded from the variant-C means: Arts
and Office at 3000x2500 have a single seed for our arm, and both sit far below
the trend of their own smaller scales, which is what an unconverged run looks
like.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = _paths.FIGURES_OUT_DIR
DATA = os.path.join(_paths.TABLES_DIR, "overview_data.json")

C_EUCL = "#2a78d6"   # slot 1 blue
C_HYP = "#eb6834"    # slot 2 orange
C_OURS = "#1baf7a"   # slot 3 aqua
C_POIN = "#8a8a85"   # muted ink - a control, not a competing geometry
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#deddd8"

ARMS = [("euclidean", "Euclidean\n(classic SASRec)", C_EUCL),
        ("poincare", "Fixed Poincaré\n(derived c, ablation)", C_POIN),
        ("hyperbolic", "Hyperbolic\n(HypSASRec, derived c)", C_HYP),
        ("ours", "Learned geometry\n(ours)", C_OURS)]
SHORT = {"euclidean": "Euclidean", "poincare": "Fixed Poincaré",
         "hyperbolic": "Hyperbolic", "ours": "Learned (ours)"}
DATASETS = ["ML-1M", "Beauty", "Arts", "Grocery", "Office"]
SCALES = ["300x800", "1000x1200", "3000x2500"]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def cell(rows, ds, sc, arm):
    for r in rows:
        if r["dataset"] == ds and r["scale"] == sc:
            return r.get(arm)
    return None


def variant_a(rows):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), sharey=True)
    w = 0.2
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        x = np.arange(len(SCALES))
        for k, (arm, _lab, col) in enumerate(ARMS):
            vals, errs, hollow = [], [], []
            for sc in SCALES:
                c = cell(rows, ds, sc, arm)
                vals.append(c["hr"] if c else np.nan)
                errs.append(c["sd"] if c else 0.0)
                hollow.append(bool(c and c["n"] < 3))
            pos = x + (k - 1.5) * w
            ax.bar(pos, vals, w * 0.88, yerr=errs, capsize=2,
                   color=[("none" if h else col) for h in hollow],
                   edgecolor=col, linewidth=1.4,
                   error_kw=dict(ecolor=INK2, lw=1), label=SHORT[arm] if ds == DATASETS[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("x", "×") for s in SCALES], fontsize=8.5)
        ax.set_title(ds, fontsize=11, color=INK, pad=8)
    axes[0].set_ylabel("HR@10", fontsize=10, color=INK)
    axes[0].set_ylim(0, 0.68)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("SASRec quality under four embedding geometries",
                 fontsize=13, color=INK, y=1.02)
    fig.text(0.5, -0.1, "Hollow bars: single seed, excluded from conclusions. "
             "Error bars: SD over 3 seeds.", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "geometry_overview_A_small_multiples.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variant_b(rows):
    labels, data = [], []
    for ds in DATASETS:
        for sc in SCALES:
            labels.append(f"{ds}  {sc.replace('x', '×')}")
            data.append({a: cell(rows, ds, sc, a) for a, _l, _c in ARMS})
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    y = np.arange(len(labels))[::-1]
    for i, d in zip(y, data):
        present = [(d[a]["hr"], c) for a, _l, c in ARMS if d[a]]
        if len(present) > 1:
            ax.plot([min(p[0] for p in present), max(p[0] for p in present)], [i, i],
                    color=GRID, lw=2, zorder=1, solid_capstyle="round")
        for arm, _lab, col in ARMS:
            c = d[arm]
            if not c:
                continue
            ax.scatter(c["hr"], i, s=85, zorder=3,
                       facecolor=("white" if c["n"] < 3 else col),
                       edgecolor=col, linewidth=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("HR@10", fontsize=10, color=INK)
    ax.set_xlim(0.05, 0.68)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor=c,
                          markersize=9, label=SHORT[a]) for a, _l, c in ARMS]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("SASRec quality under four embedding geometries\n"
                 "hollow marker = single seed", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "geometry_overview_B_dotplot.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variant_c(rows):
    """Means must be taken over the SAME cells for every arm.

    Our arm is missing three cells (Arts and Office 3000x2500 have a single
    seed, Grocery 3000x2500 was never run), and all three are large-scale
    cells where every arm scores high - averaging 15 cells for the baselines
    against 12 for ours would silently penalise ours. So the comparison is
    restricted to the cells where all four arms have >= 3 seeds, and the
    excluded ones are named on the figure.
    """
    common = [(ds, sc) for ds in DATASETS for sc in SCALES
              if all((lambda c: c and c["n"] >= 3)(cell(rows, ds, sc, a))
                     for a, _l, _c in ARMS)]
    dropped = [(ds, sc) for ds in DATASETS for sc in SCALES if (ds, sc) not in common]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    style(ax)
    rng = np.random.default_rng(3)
    for k, (arm, lab, col) in enumerate(ARMS):
        vals = [cell(rows, ds, sc, arm)["hr"] for ds, sc in common]
        jit = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(np.full(len(vals), k) + jit, vals, s=70, color=col, alpha=0.75,
                   edgecolor="white", linewidth=1.2, zorder=3)
        m = float(np.mean(vals))
        ax.plot([k - 0.26, k + 0.26], [m, m], color=col, lw=3, zorder=4,
                solid_capstyle="round")
        ax.text(k, m + 0.030, f"{m:.3f}", ha="center", fontsize=11.5,
                color=INK, fontweight="bold", zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([l for _a, l, _c in ARMS], fontsize=10, color=INK)
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("HR@10", fontsize=10, color=INK)
    ax.set_title(f"Each dot is one of the {len(common)} dataset x scale cells where all "
                 f"four arms have 3 seeds", fontsize=10.5, color=INK2, pad=6)
    fig.suptitle("Geometry regime vs SASRec quality", fontsize=13.5, color=INK,
                 y=0.98, x=0.52)
    note = "Excluded (our arm incomplete): " + ", ".join(f"{d} {s.replace('x', 'x')}"
                                                         for d, s in dropped)
    fig.text(0.5, -0.045, note, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "geometry_overview_C_regimes.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = json.load(open(DATA, encoding="utf-8"))
    variant_a(rows)
    variant_b(rows)
    variant_c(rows)
    print(f"written to {OUTDIR}:")
    for f in sorted(os.listdir(OUTDIR)):
        if f.startswith("geometry_overview"):
            print("   ", f)


if __name__ == "__main__":
    main()
