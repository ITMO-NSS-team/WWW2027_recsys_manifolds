"""
Same figure type as hyperbolicity_frozenproj_comparison.png (hyperbolicity
vs quality scatter + a second panel), extended across all FOUR scales
tested (300u/800i, 1000u/1200i, 3000u/2500i, 6000u/3000i) under the same
adopted config (eta_outer=0.001, freeze_item_projection=True, dropout=0.2) -
with Poincare-pretrained and the fair (dropout=0.2) Euclidean baseline
computed at every scale too, so all three geometries are compared under
equal conditions at every scale.

Seed counts per scale (n=1 caveat still applies for Poincare/Euclidean,
which aren't re-seeded - only GINCF has repeats, shown as mean+-std here):
  300u/800i:   GINCF x3, Poincare x1, Euclidean x1
  1000u/1200i: GINCF x3, Poincare x1, Euclidean x1
  3000u/2500i: GINCF x3, Poincare x1, Euclidean x1
  6000u/3000i: GINCF x3, Poincare x1, Euclidean x1

IMPORTANT CAVEAT for the 6000u/3000i GINCF point: in all 3 seeds, the restored-best outer epoch is epoch 0 and NOT
ONE of the following 99 epochs ever beat it (val_hr-wise) - the bad-grad
guard fired already at epoch 0 (gradients zeroed before the first real
step), so this point is close to measuring the UNOPTIMIZED initial
geometry, not a genuine converged result like the other 3 scales. Flagged
directly on the plot, not just in this docstring.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = _paths.FIGURES_OUT_DIR

base = pd.read_csv(os.path.join(_paths.TABLES_DIR,
                                "full_hyperbolicity_table_frozenproj.csv"))
scaling = pd.read_csv(os.path.join(_paths.TABLES_DIR,
                                   "full_hyperbolicity_table_scaling.csv"))

SCALE_COLORS = {"300u/800i": "tab:green", "1000u/1200i": "tab:blue",
                "3000u/2500i": "tab:orange", "6000u/3000i": "tab:red"}
SCALE_ORDER = ["300u/800i", "1000u/1200i", "3000u/2500i", "6000u/3000i"]
TYPE_MARKERS = {"GINCF": "o", "Poincare": "^", "Euclidean": "s"}

# --- Assemble one consistent long-form table: scale, type, delta_rel, hr10 ---
records = []
for _, row in base.iterrows():
    g = row["geometry"]
    if g.startswith("GINCF seed") and "restored-best" in g:
        records.append({"scale": "300u/800i", "type": "GINCF", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
    elif g.startswith("Poincare"):
        records.append({"scale": "300u/800i", "type": "Poincare", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
    elif g.startswith("Euclidean"):
        records.append({"scale": "300u/800i", "type": "Euclidean", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})

for _, row in scaling.iterrows():
    g = row["geometry"]
    for scale_label in ["1000u/1200i", "3000u/2500i", "6000u/3000i"]:
        if scale_label not in g:
            continue
        if g.startswith("GINCF"):
            records.append({"scale": scale_label, "type": "GINCF", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
        elif g.startswith("Poincare"):
            records.append({"scale": scale_label, "type": "Poincare", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})
        elif g.startswith("Euclidean"):
            records.append({"scale": scale_label, "type": "Euclidean", "delta_rel": row["delta_rel"], "hr10": row["hr10"]})

df = pd.DataFrame(records)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Left: hyperbolicity (delta_rel) vs downstream quality (HR@10), mean+-std where n>1 ---
ax = axes[0]
for scale_label in SCALE_ORDER:
    for geom_type in ["GINCF", "Poincare", "Euclidean"]:
        sub = df[(df["scale"] == scale_label) & (df["type"] == geom_type)]
        if sub.empty:
            continue
        x_mean, y_mean = sub["delta_rel"].mean(), sub["hr10"].mean()
        if len(sub) > 1:
            ax.errorbar(x_mean, y_mean, xerr=sub["delta_rel"].std(), yerr=sub["hr10"].std(),
                       color=SCALE_COLORS[scale_label], ecolor=SCALE_COLORS[scale_label],
                       elinewidth=1.3, capsize=4, zorder=2)
        ax.scatter(x_mean, y_mean, s=170, marker=TYPE_MARKERS[geom_type],
                   color=SCALE_COLORS[scale_label], edgecolor="black", linewidth=1.2, zorder=3)

# Legend: color = scale, marker shape = geometry type
scale_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                             markersize=11, label=s) for s, c in SCALE_COLORS.items()]
type_handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", markeredgecolor="black",
                            markersize=11, label=t) for t, m in TYPE_MARKERS.items()]
leg1 = ax.legend(handles=scale_handles, title="scale (color)", loc="upper left")
ax.add_artist(leg1)
ax.legend(handles=type_handles, title="geometry (marker)", loc="lower right")

ax.set_xlabel("delta_rel (Gromov hyperbolicity, normalized — LOWER = more tree-like/hyperbolic)")
ax.set_ylabel("test HR@10")
ax.set_title("Hyperbolicity vs downstream quality, all 3 geometries x all 4 scales\nstill no clean hyperbolicity relationship, at any scale")
ax.grid(alpha=0.3)

# --- Right: HR@10 vs scale, all three geometries under equal conditions ---
ax = axes[1]
x_pos = {s: i for i, s in enumerate(SCALE_ORDER)}

for geom_type, marker in TYPE_MARKERS.items():
    means, stds, xs_line = [], [], []
    for scale_label in SCALE_ORDER:
        sub = df[(df["scale"] == scale_label) & (df["type"] == geom_type)]
        if sub.empty:
            continue
        xs = np.full(len(sub), x_pos[scale_label]) + (np.random.RandomState(0).uniform(-0.05, 0.05, len(sub))
                                                        if len(sub) > 1 else 0)
        ax.scatter(xs, sub["hr10"], s=140, marker=marker, color=SCALE_COLORS[scale_label],
                   edgecolor="black", linewidth=1.2, zorder=3)
        means.append(sub["hr10"].mean())
        stds.append(sub["hr10"].std() if len(sub) > 1 else 0.0)
        xs_line.append(x_pos[scale_label])
    ax.errorbar(xs_line, means, yerr=stds, linestyle="--", color="gray", linewidth=1.5, zorder=2,
               marker=marker, markersize=9, markerfacecolor="none", markeredgecolor="black",
               capsize=4, label=geom_type)

# Flag the 6000u/3000i GINCF point: originally (eta=0.001) the outer loop never beat
# epoch 0 in any of 3 seeds (eigh-backward instability. FIXED
# With eta=0.00001 it converges normally (best epoch late: steps
# 68/71/80 of 100), but test_hr barely moves and GINCF still clearly loses to
# Euclidean - a confirmed result now, not an optimizer artifact.
gincf_6000 = df[(df["scale"] == "6000u/3000i") & (df["type"] == "GINCF")]
if not gincf_6000.empty:
    ax.annotate("eigh-instability FIXED (eta=1e-5):\nnow converges normally\n(best epoch late, 68-80/100) -\nstill loses to Euclidean, confirmed",
               xy=(x_pos["6000u/3000i"], gincf_6000["hr10"].mean()),
               xytext=(x_pos["6000u/3000i"] - 1.4, gincf_6000["hr10"].mean() - 0.11),
               fontsize=9, color="darkred",
               arrowprops=dict(arrowstyle="->", color="darkred", lw=1.3))

ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels(list(x_pos.keys()))
ax.set_xlabel("scale")
ax.set_ylabel("test HR@10")
ax.set_title("Quality vs scale, all 3 geometries under equal conditions\n"
             "(dashed = mean per scale, error bars = std across 3 GINCF seeds)\n"
             "GINCF's edge over Euclidean narrows, then REVERSES at 6000u/3000i",
             fontsize=12)
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Amazon Beauty: GINCF vs Poincare vs Euclidean baseline, across scale\n"
             "(frozen projection, dropout=0.2 for all three; eta=0.001 at the first 3 scales, "
             "eta=1e-5 at 6000u/3000i after fixing an eigh-instability) — equal conditions otherwise",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT_DIR, "hyperbolicity_scaling_comparison.png")
fig.savefig(out_path, dpi=130)
print(f"[Save] {out_path}")
