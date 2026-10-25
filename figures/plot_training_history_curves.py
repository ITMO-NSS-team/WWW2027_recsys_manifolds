"""Quality saturation vs geometry movement, per outer epoch.

Produces two figures - one per dataset - from the per-epoch histories in the
run logs. Nothing here reads a snapshot: the logs carry every epoch, while the
diagnostics CSVs sample every twentieth.

  np_amazon_history.png   Amazon Beauty: bary-cos, ItemKNN
  np_ml1m_history.png     MovieLens-1M:  bary-cos, ItemKNN, GI-NCF

GI-NCF appears only on ml-1m because every GI-NCF run is on ml-1m.

WHICH GI-NCF RUNS. Several NCF trees exist in EXTERNAL_RUNS_DIR/GINCF; this
figure uses `patience_300`, the paper configuration (lr_isomap=0.001,
freeze_item_projection=True, 300 outer epochs, cf_epochs=200), so that NCF
shares both a horizon and a learning-rate regime with the np heads. The other
trees are hyperparameter searches and are not used anywhere in this
repository.

WHY SASREC IS ABSENT. GI-SASRec writes `history.json` instead of a run log, and
that file has val_hr, val_ndcg and grad_nonfinite_count but no
`geom_delta_init`. The bottom row cannot be drawn for it from logs alone. It is
computable - all 300 snapshots are saved - but that is a separate pass over the
matrices, not a log read.

Two marked epochs, which mean different things:

  marker + dashed line   `best_epoch` as recorded, i.e. the argmax of a noisy
                         validation curve; on a plateau it lands arbitrarily
  pale solid line        saturation: first epoch at which a 21-epoch centred
                         rolling mean reaches within 1% of its own maximum and
                         stays there

Two rows rather than one panel with two y-scales: a dual-axis chart would make
the crossing point of the curves an artefact of the scaling, and where each
curve flattens relative to the other is the whole point.
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
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")
PAPER = _paths.FIGURES_OUT_DIR
SHARE = _paths.EXTERNAL_RUNS_DIR
ML1M_NP = os.path.join(SHARE, "GI_np_baselines_ml-1m")
NCF_PAPER = os.path.join(SHARE, "GINCF", "patience_300")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#deddd8"
SCALES = ["300x800", "1000x1200", "3000x2500"]
STYLE = {"bary_cos": ("bary-cos", "#2a78d6", "o"),
         "item_knn": ("ItemKNN", "#eb6834", "D"),
         "gi_ncf": ("GI-NCF", "#00702f", "s")}

# ml-1m np cells, copied verbatim from compute_diagnostics_nonparametric_heads_ml1m.py so this figure and
# the already-computed ml-1m diagnostics describe the same runs
ML1M_CELLS = {
    ("bary_cos", "300x800"):   r"bary_cos\lr_1e-2\n_neighbors_10\300_800_seed_0",
    ("bary_cos", "1000x1200"): r"bary_cos\lr_1e-2\n_neighbors_10\1000_1200_seed_0",
    ("bary_cos", "3000x2500"): r"bary_cos\lr_1e-2\n_neighbors_40\3000_2500_seed_0",
    ("item_knn", "300x800"):   r"item_knn\nn_10_lr_1e-3\knn_k_20_temp_0.5\300_800_seed_0",
    ("item_knn", "1000x1200"): r"item_knn\nn_10_lr_1e-3\knn_k_20_temp_0.5\1000_1200_seed_0",
    ("item_knn", "3000x2500"): r"item_knn\nn_40_lr_1e-2\knn_k_20_temp_0.5\3000_2500_seed_0",
}


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
    """First epoch at which the smoothed curve reaches within tol of its max.

    Read as "stopped improving", not "settled and stayed". An earlier version
    additionally required the curve never to fall back below the threshold,
    which returned None for every declining run - and declining is exactly what
    ItemKNN does. A curve that peaks early and then decays has stopped
    improving at its peak; that is the epoch worth marking.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return None
    sm = np.convolve(y, np.ones(window) / window, mode="same")
    edge = window // 2
    sm[:edge] = np.nan
    sm[-edge:] = np.nan
    ok = ~np.isnan(sm)
    if ok.sum() < 3:
        return None
    hit = np.where(ok & (sm >= np.nanmax(sm) * (1 - tol)))[0]
    return int(hit[0]) if len(hit) else None


def history_from_log(run_dir, section):
    """val_hr, geom_delta_init, best_epoch.

    The np logs record `best_epoch`; the GI-NCF `gi` block does not, so there
    it falls back to the argmax of val_hr. Both are the same quantity - the
    argmax of a noisy curve - so the marker means the same thing either way.
    """
    with open(os.path.join(run_dir, "run_log.json")) as f:
        blk = json.load(f)[section]
    h = blk["history"]
    hr = np.asarray(h["val_hr"], dtype=float)
    best = blk.get("best_epoch")
    if best is None and len(hr):
        best = int(np.nanargmax(hr))
    return hr, np.asarray(h["geom_delta_init"], dtype=float), best


def collect(dataset):
    """-> {(series, scale): (val_hr, geom_delta_init, best_epoch)}"""
    out = {}
    if dataset == "amazon":
        inv = pd.read_csv(INVENTORY)
        fin = inv[inv.status == "finished"]
        for (head, scale), g in fin.groupby(["head", "scale"]):
            combo = g.groupby("combo").hr.mean().idxmax()
            r = g[(g.combo == combo) & (g.seed == 0)]
            if not r.empty:
                out[(head, scale)] = history_from_log(r.iloc[0].run_dir,
                                                      "geometry_baseline")
    else:
        for (head, scale), sub in ML1M_CELLS.items():
            d = os.path.join(ML1M_NP, sub)
            if os.path.isdir(d):
                out[(head, scale)] = history_from_log(d, "geometry_baseline")
        for scale in SCALES:
            d = os.path.join(NCF_PAPER, scale.replace("x", "_") + "_seed_0")
            if os.path.isdir(d):
                out[("gi_ncf", scale)] = history_from_log(d, "gi")
    return out


def render(dataset, title, fname):
    data = collect(dataset)
    if not data:
        print(f"[{dataset}] no data")
        return
    series = [s for s in STYLE if any(k[0] == s for k in data)]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.0), sharex=True)
    rows = []
    for col, scale in enumerate(SCALES):
        for row in (0, 1):
            style(axes[row, col])
        for s in series:
            got = data.get((s, scale))
            if got is None:
                continue
            hr, gd, best = got
            lab, c, mk = STYLE[s]
            ep = np.arange(len(hr))
            axes[0, col].plot(ep, hr, color=c, lw=1.3, alpha=0.9,
                              label=lab if col == 0 else None)
            axes[1, col].plot(ep, gd, color=c, lw=1.3, alpha=0.9)

            sat = saturation_epoch(hr)
            if sat is not None:
                for ax in (axes[0, col], axes[1, col]):
                    ax.axvline(sat, color=c, lw=1.8, alpha=0.28, zorder=1)
            if best is not None and best < len(hr):
                for ax, y in ((axes[0, col], hr), (axes[1, col], gd)):
                    ax.scatter([best], [y[best]], s=44, color=c, marker=mk,
                               edgecolor=INK, linewidth=0.8, zorder=5)
                    ax.axvline(best, color=c, lw=0.9, ls=(0, (4, 3)),
                               alpha=0.45, zorder=1)
            rows.append((lab, scale, best, sat, hr, gd))
        axes[0, col].set_title(scale, fontsize=11, color=INK, pad=6)
        axes[1, col].set_xlabel("outer epoch", fontsize=9.5, color=INK2)

    axes[0, 0].set_ylabel("val HR@10", fontsize=10.5, color=INK)
    axes[1, 0].set_ylabel("geometry distance\nfrom initialisation",
                          fontsize=10.5, color=INK)

    # the series entries carry their own marker, so the legend explains the
    # markers without a separate "what the dot means" line
    handles = [plt.Line2D([], [], color=STYLE[s][1], lw=1.6,
                          marker=STYLE[s][2], markersize=7,
                          markeredgecolor=INK, markeredgewidth=0.8)
               for s in series]
    labels = [f"{STYLE[s][0]} — line, marker at its best epoch" for s in series]
    handles += [plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3))),
                plt.Line2D([], [], color=INK2, lw=3.4, alpha=0.28)]
    labels += ["best epoch (argmax of the noisy curve, as logged)",
               "saturation (21-epoch mean within 1% of its max)"]
    ncol = 3 if len(handles) > 4 else 2
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9.5, columnspacing=1.8, handletextpad=0.7,
               bbox_to_anchor=(0.5, -0.135 if ncol == 3 else -0.10))
    fig.suptitle(title, fontsize=12.5, color=INK, y=1.005)
    fig.tight_layout(w_pad=0.6)
    out = os.path.join(PAPER, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Save] {out}")

    print(f'\n{title}')
    print(f'{"series":10s} {"scale":11s} {"best":>5s} {"satur":>6s} '
          f'{"HR(best)":>9s} {"HR(last)":>8s} {"shift(satur)":>13s} '
          f'{"shift(last)":>11s} {"after satur":>12s}')
    for lab, scale, best, sat, hr, gd in rows:
        gs = gd[sat] if sat is not None else float("nan")
        extra = (gd[-1] - gs) / gd[-1] if gd[-1] else float("nan")
        print(f"{lab:10s} {scale:11s} "
              f"{best if best is not None else -1:5d} "
              f"{sat if sat is not None else -1:6d} "
              f"{hr[best] if best is not None else float('nan'):9.4f} "
              f"{hr[-1]:8.4f} {gs:13.4f} {gd[-1]:11.4f} {extra:11.1%}")


def main():
    render("amazon", "Amazon Beauty — non-parametric heads",
           "np_amazon_history.png")
    render("ml1m", "MovieLens-1M — non-parametric heads and GI-NCF",
           "np_ml1m_history.png")


if __name__ == "__main__":
    main()
