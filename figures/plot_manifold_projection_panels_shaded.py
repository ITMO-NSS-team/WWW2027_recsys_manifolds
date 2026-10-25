"""Four-panel item clouds with a soft density backing behind the points.

Same four geometries and same projection as plot_manifold_projection_panels.py; what is added
is a soft shaded backdrop behind the points and a count of how many separated
regions each geometry actually has.

Why not kNN for the neighbourhood structure: a k-nearest-neighbour graph assumes
a roughly isotropic neighbourhood, and the learned manifolds here are full of
long thin arcs (Office is the clearest case). Any fixed k either chops an arc
into fragments or bridges across the gap between two arcs. Both tools used here
are shape-agnostic instead:

  * the backdrop is a kernel density estimate on a grid, filled above a
    percentile threshold - it has no notion of cluster shape at all, so an arc
    stays an arc;
  * the region count comes from HDBSCAN on the 2-D coordinates, which grows
    clusters by density connectivity rather than by a neighbour count. Checked
    on a synthetic arc-plus-blob mixture: 2 clusters, 0 points dropped as noise.

The backdrop is a single neutral tint, never tinted per cluster: the points
already carry a diverging red-blue ramp for popularity, and a second colour
channel behind them would collide with it.

"N regions" is the honest form of the claim "the space becomes more locally
defined" - a number from the data rather than an impression from the picture.
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
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_manifold_projection_panels import (  # noqa: E402
    DATASETS, SCALES, SEED, INK, INK2, GRID, collect, load_D, popularity,
)

OUTDIR = os.path.join(_paths.FIGURES_OUT_DIR, "manifold_panels4_shaded")
SHADE = "#b9b9b1"


def density_backdrop(ax, xy, bins=200, sigma=3.5, level_q=58):
    """Filled KDE island(s) behind the points. Shape-agnostic by construction."""
    x, y = xy[:, 0], xy[:, 1]
    xr, yr = x.max() - x.min(), y.max() - y.min()
    pad = 0.07
    ext = [x.min() - pad * xr, x.max() + pad * xr,
           y.min() - pad * yr, y.max() + pad * yr]
    H, xe, ye = np.histogram2d(x, y, bins=bins,
                               range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    H = gaussian_filter(H.T, sigma=sigma)
    if H.max() <= 0:
        return ext
    thr = np.percentile(H[H > 0], level_q)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(xc, yc, H, levels=[thr, H.max() * 1.01], colors=[SHADE],
                alpha=0.42, zorder=1)
    ax.contour(xc, yc, H, levels=[thr], colors=[SHADE], linewidths=0.8,
               alpha=0.9, zorder=2)
    return ext


def n_regions(xy):
    """HDBSCAN region count: density-connected, so arcs survive as one region."""
    n = len(xy)
    mcs = max(8, int(n / 120))
    lab = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(xy)
    k = len(set(lab)) - (1 if -1 in lab else 0)
    noise = float(np.mean(lab == -1))
    return k, noise, mcs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # region counts still computed and written to region_counts.json; the figures
    # carry no sub-captions at all, by request - the shading is decorative there.
    made, skipped, summary = [], [], []
    for ds_name, tag, dirname, category in DATASETS:
        for (N, M) in SCALES:
            panels = collect(tag, N, M)
            miss = [p[0] for p in panels if not p[1]]
            if miss:
                skipped.append(f"{ds_name} {N}x{M}: missing {', '.join(miss)}")
                continue
            t0 = time.time()
            pop = popularity(dirname, category, N, M)
            fig, axes = plt.subplots(1, 4, figsize=(19.5, 6.1))
            sc, row = None, {"dataset": ds_name, "scale": f"{N}x{M}"}
            for ax, (title, path, cap, met) in zip(axes, panels):
                D = load_D(path)
                n = min(len(pop), D.shape[0])
                D, c = D[:n, :n], pop[:n]
                med = max(np.median(c), 1.0)
                cval = np.log2(np.maximum(c, 0.5) / med)
                lim = float(np.percentile(np.abs(cval), 98)) or 1.0
                xy = TSNE(n_components=2, metric="precomputed", init="random",
                          perplexity=30, random_state=0, max_iter=1000).fit_transform(D)

                ext = density_backdrop(ax, xy)
                k, noise, mcs = n_regions(xy)
                row[title] = k
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=cval, s=9, cmap="RdBu_r",
                                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                                linewidth=0, alpha=0.9, zorder=3)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_aspect("equal", adjustable="box")
                ax.set_facecolor("#fcfcfb")
                for s in ax.spines.values():
                    s.set_color(GRID)
                ax.set_xticks([]); ax.set_yticks([])
                sub = (f"HR@10 {met[0]:.3f}  ·  NDCG@10 {met[1]:.3f}" if met
                       else "no model trained")
                ax.set_title(f"{title}\n{sub}", fontsize=11.5, color=INK, pad=8)
                # no sub-caption at all, by request: the shading is decorative here.
                # Region counts still go to region_counts.json.

            cb = fig.colorbar(sc, ax=axes, fraction=0.011, pad=0.01)
            cb.set_label("popularity vs median item  (log₂ ratio)", fontsize=9, color=INK2)
            cb.ax.tick_params(colors=INK2, labelsize=8)
            cb.outline.set_edgecolor(GRID)
            fig.suptitle(f"{ds_name} — {N}×{M}, seed {SEED}: the same {n} items, "
                         f"four geometries", fontsize=13.5, color=INK, y=0.99)
            out = os.path.join(OUTDIR, f"manifold4sh_{tag}_{N}x{M}.png")
            fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            made.append(os.path.basename(out))
            summary.append(row)
            print(f"[ok] {ds_name} {N}x{M}: " +
                  "  ".join(f"{t.split()[0]}={row[t]}" for t in row if t not in ("dataset", "scale")) +
                  f"   ({time.time()-t0:.0f}s)", flush=True)

    json.dump(summary, open(os.path.join(OUTDIR, "region_counts.json"), "w"), indent=1)
    print(f"\n=== {len(made)} figures in {OUTDIR} ===")
    for s in skipped:
        print(f"   skipped - {s}")


if __name__ == "__main__":
    main()
