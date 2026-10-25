"""Quality saturation vs geometry movement, per epoch, for the np baselines.

Both curves come straight out of `run_log.json` -> `geometry_baseline.history`,
which stores 300 entries per run. Nothing here needs the snapshots: the
diagnostics CSVs sample every 20th epoch, the log has every epoch, and the two
questions this figure answers - when did quality stop improving, and when did
the geometry stop moving - are both answered at epoch resolution by the log.

Layout: two rows over three scale columns.

  top     val HR@10 against epoch, one line per head, with a marker at the
          best epoch recorded in the log
  bottom  `geom_delta_init`, the distance of the current geometry from the
          initial approximation, same axes

Two rows rather than one panel with two y-scales: a dual-axis chart would make
the crossing point of the two curves an artefact of the scaling, and the whole
point of the figure is *where* each curve flattens relative to the other.

Two different epochs are marked, because they mean different things.

`best_epoch` is what the run log records, and it is the argmax of a visibly
noisy validation curve. On a run that rises and then plateaus, the argmax lands
at an arbitrary point inside the plateau - bary-cos at 1000x1200 reports 271 of
300 even though the curve has been flat since roughly 150. Quoting it as "the
epoch where it saturated" would overstate what it is.

So saturation is marked separately and defined explicitly: the first epoch at
which a 21-epoch centred rolling mean reaches within 1% of its own maximum, and
stays there. That is a property of the smoothed trend rather than of a single
noisy sample.

Both are marked rather than used as endpoints. The before/after tables take
"after" to be the final epoch, and on several runs quality peaked far earlier.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHARE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baseline_amazon")
HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")
PAPER = _paths.FIGURES_OUT_DIR

INK, INK2, GRID = "#0b0b0b", "#52514e", "#deddd8"
HEADS = [("bary_cos", "bary-cos", "#2a78d6", "o"),
         ("item_knn", "ItemKNN", "#eb6834", "D")]
SCALES = ["300x800", "1000x1200", "3000x2500"]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def saturation_epoch(y, window=21, tol=0.01):
    """First epoch where the smoothed curve is within `tol` of its max and stays.

    Returns None if the curve never settles - which is itself informative and
    must not be silently reported as "saturated at the last epoch".
    """
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return None
    k = np.ones(window) / window
    sm = np.convolve(y, k, mode="same")
    edge = window // 2
    sm[:edge] = np.nan
    sm[-edge:] = np.nan
    ok = ~np.isnan(sm)
    if ok.sum() < 3:
        return None
    thr = np.nanmax(sm) * (1 - tol)
    above = np.where(ok & (sm >= thr))[0]
    if not len(above):
        return None
    # first index from which the smoothed curve never drops below the threshold
    idx = np.where(ok)[0]
    for i in above:
        tail = sm[idx[idx >= i]]
        if np.all(tail >= thr * 0.995):
            return int(i)
    return int(above[0])


def load_history(run_dir):
    with open(os.path.join(run_dir, "run_log.json")) as f:
        gb = json.load(f)["geometry_baseline"]
    return gb["history"], gb.get("best_epoch")


def main():
    import pandas as pd
    inv = pd.read_csv(INVENTORY)
    fin = inv[inv.status == "finished"]

    # same cell rule as the diagnostics: winning combo by mean HR, seed 0
    picks = {}
    for (head, scale), g in fin.groupby(["head", "scale"]):
        combo = g.groupby("combo").hr.mean().idxmax()
        r = g[(g.combo == combo) & (g.seed == 0)]
        if not r.empty:
            picks[(head, scale)] = r.iloc[0].run_dir

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.0), sharex=True)
    best_rows = []
    for col, scale in enumerate(SCALES):
        for row in (0, 1):
            style(axes[row, col])
        for head, lab, c, mk in HEADS:
            rd = picks.get((head, scale))
            if rd is None:
                continue
            h, best = load_history(rd)
            ep = np.asarray(h["epoch"])
            hr = np.asarray(h["val_hr"])
            gd = np.asarray(h["geom_delta_init"])

            axes[0, col].plot(ep, hr, color=c, lw=1.4, alpha=0.9,
                              label=lab if col == 0 else None)
            axes[1, col].plot(ep, gd, color=c, lw=1.4, alpha=0.9)
            sat = saturation_epoch(hr)
            if sat is not None:
                for ax, series in ((axes[0, col], hr), (axes[1, col], gd)):
                    ax.axvline(sat, color=c, lw=1.6, alpha=0.30, zorder=1)
            if best is not None:
                axes[0, col].scatter([best], [hr[best]], s=46, color=c,
                                     marker=mk, edgecolor=INK, linewidth=0.8,
                                     zorder=5)
                axes[1, col].scatter([best], [gd[best]], s=46, color=c,
                                     marker=mk, edgecolor=INK, linewidth=0.8,
                                     zorder=5)
                for ax in (axes[0, col], axes[1, col]):
                    ax.axvline(best, color=c, lw=0.9, ls=(0, (4, 3)), alpha=0.5,
                               zorder=1)
                # how much of the geometry's total travel happened after the
                # metric peaked - the number the figure exists to show
                best_rows.append((head, scale, int(best), sat, float(hr[best]),
                                  float(hr[-1]), float(gd[best]), float(gd[-1]),
                                  float(gd[sat]) if sat is not None else float("nan")))
        axes[0, col].set_title(scale, fontsize=11, color=INK, pad=6)
        axes[1, col].set_xlabel("outer epoch", fontsize=9.5, color=INK2)

    axes[0, 0].set_ylabel("val HR@10", fontsize=10.5, color=INK)
    axes[1, 0].set_ylabel("geometry distance\nfrom initialisation",
                          fontsize=10.5, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3))))
    labels.append("best epoch (argmax, as logged)")
    handles.append(plt.Line2D([], [], color=INK2, lw=3.2, alpha=0.30))
    labels.append("saturation (smoothed curve within 1% of its max)")
    # ncol=4 keeps the legend on one line; at ncol=3 the fourth entry wrapped
    # onto a second row and sat on top of the "outer epoch" labels
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, columnspacing=1.6, handletextpad=0.6,
               bbox_to_anchor=(0.5, -0.075))
    fig.tight_layout(w_pad=0.6)
    out = os.path.join(PAPER, "np_amazon_history.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[Save] {out}")

    print("\nwhat happens after quality stops improving:")
    print(f'{"head":10s} {"scale":11s} {"best":>5s} {"satur":>6s} '
          f'{"HR(best)":>9s} {"HR(300)":>8s} {"shift(satur)":>13s} '
          f'{"shift(300)":>11s} {"extra":>8s}')
    for head, scale, best, sat, hb, hl, gb_, gl, gs in best_rows:
        extra = (gl - gs) / gl if (gl and gs == gs) else float("nan")
        sat_s = f"{sat:6d}" if sat is not None else "     -"
        print(f"{head:10s} {scale:11s} {best:5d} {sat_s} {hb:9.4f} {hl:8.4f} "
              f"{gs:13.4f} {gl:11.4f} {extra:7.1%}")


if __name__ == "__main__":
    main()
