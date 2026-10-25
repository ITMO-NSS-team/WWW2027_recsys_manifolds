"""MovieLens-1M: clusters found in the ORIGINAL space, drawn on the 2-D projection.

The earlier version clustered the t-SNE output, which makes the regions a
property of the projection rather than of the geometry - t-SNE both splits and
merges. Here HDBSCAN runs on each panel's own distance matrix
(`metric="precomputed"`), i.e. in the space that geometry actually lives in -
the raw profile space for the data panel, the 64-dim embedding table for the
Euclidean one, the 32-dim tangent coordinates for Poincare, the 32-dim latent
for ours. Only the *display* is 2-D: each cluster's backdrop is a density field
over the 2-D positions of its own members.

A consequence worth expecting: clusters found in the original space can look
torn apart in the projection, because t-SNE is under no obligation to keep them
contiguous. That is informative rather than a bug - a backdrop that fragments
means the projection is not faithful to that geometry's own neighbourhoods.

Two figures per scale:
  *_genres.png   - backdrop = genre territory (semantic reference)
  *_clusters.png - backdrop = clusters from the original space

Points carry the popularity ramp in both, by request. That does put two colour
encodings in one panel; the backdrop is held at low alpha so the points stay
first in the reading order, but the collision is real and worth judging by eye.

AMI between the original-space clusters and primary genre is printed for every
panel - adjusted for chance, so a larger cluster count cannot inflate it.
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
import sys
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, to_rgb
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_manifold_projection_panels import SCALES, SEED, INK, INK2, GRID, collect, load_D, popularity  # noqa: E402
from plot_manifold_genre_territories_ml1m import genre_of_items, HUES, OTHER  # noqa: E402

OUTDIR = os.path.join(_paths.FIGURES_OUT_DIR, "manifold_panels4_genres")


def territory(ax, xy, labels, order, colors, bins=170, sigma=4.0, q=55, alpha=0.26):
    """Tint each grid cell by whichever label dominates its neighbourhood."""
    x, y = xy[:, 0], xy[:, 1]
    xr, yr = x.max() - x.min(), y.max() - y.min()
    pad = 0.07
    ext = [x.min() - pad * xr, x.max() + pad * xr, y.min() - pad * yr, y.max() + pad * yr]
    rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
    fields, tot = [], None
    for g in order:
        sel = labels == g
        if sel.sum() == 0:
            H = np.zeros((bins, bins))
        else:
            H, xe, ye = np.histogram2d(x[sel], y[sel], bins=bins, range=rng)
            H = gaussian_filter(H.T, sigma=sigma)
        fields.append(H)
        tot = H.copy() if tot is None else tot + H
    fields = np.array(fields)
    win = fields.argmax(axis=0)
    thr = np.percentile(tot[tot > 0], q) if (tot > 0).any() else 0.0
    rgba = np.zeros(win.shape + (4,))
    for k, g in enumerate(order):
        m = (win == k) & (tot >= thr)
        rgba[m, :3] = to_rgb(colors[g])
        rgba[m, 3] = alpha
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear",
              zorder=1, aspect="auto")
    return ext


def cluster_in_original_space(D, n, k=8):
    """Average-linkage agglomerative clustering on the geometry's own distances.

    HDBSCAN was tried first and degenerates here: run on precomputed distances
    in 32-64 dimensions it pushed 72-100% of items into noise and found no
    clusters at all in seven of twelve panels, because density-based clustering
    needs a density contrast that high-dimensional point clouds of this size do
    not have. Any AMI it produced was computed over the surviving few percent of
    the catalogue and was not comparable between panels.

    Average linkage with a fixed k assigns every item, needs only a metric, and
    keeps k identical across panels so the chance-adjusted AMI is comparable.
    """
    lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                  linkage="average").fit_predict(D)
    sizes = Counter(lab)
    top = [l for l, _ in sizes.most_common(len(HUES))]
    named = np.array([f"C{l}" if l in top else "other" for l in lab])
    order = [f"C{l}" for l in top] + (["other"] if (named == "other").any() else [])
    return lab, named, order


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    for (N, M) in SCALES:
        panels = collect("ml1m", N, M)
        if any(not p[1] for p in panels):
            print(f"skip {N}x{M}", flush=True)
            continue
        t0 = time.time()
        genres, gorder, _ni = genre_of_items(N, M)
        pop = popularity("ml-1m", None, N, M)
        gcolors = {g: (HUES[i] if g != "Other" else OTHER) for i, g in enumerate(gorder)}

        cache = []
        for (title, path, cap, met) in panels:
            D = load_D(path)
            n = min(len(genres), D.shape[0])
            D = D[:n, :n]
            xy = TSNE(n_components=2, metric="precomputed", init="random",
                      perplexity=30, random_state=0, max_iter=1000).fit_transform(D)
            lab, named, corder = cluster_in_original_space(D, n)
            ami = adjusted_mutual_info_score(genres[:n], lab)
            big = Counter(lab).most_common(1)[0][1] / n
            print(f"  {N}x{M} {title:16s} AMI with genre {ami:+.3f}  "
                  f"(largest cluster {big:.0%} of the catalogue)", flush=True)
            report.append({"scale": f"{N}x{M}", "panel": title, "k": 8,
                           "largest_cluster_frac": round(float(big), 3),
                           "AMI_genre": round(float(ami), 4)})
            cache.append((title, met, xy, named, corder, n))

        for mode in ("genres", "clusters"):
            fig, axes = plt.subplots(1, 4, figsize=(19.5, 6.0))
            sc = None
            for ax, (title, met, xy, named, corder, n) in zip(axes, cache):
                if mode == "genres":
                    ext = territory(ax, xy, genres[:n], gorder, gcolors)
                else:
                    cc = {g: (HUES[i] if g not in ("other", "noise") else OTHER)
                          for i, g in enumerate(corder)}
                    ext = territory(ax, xy, named, corder, cc)
                c = pop[:n]
                med = max(np.median(c), 1.0)
                cval = np.log2(np.maximum(c, 0.5) / med)
                lim = float(np.percentile(np.abs(cval), 98)) or 1.0
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=cval, s=8, cmap="RdBu_r",
                                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                                linewidth=0, alpha=0.92, zorder=3)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_aspect("equal", adjustable="box")
                ax.set_facecolor("#fcfcfb")
                for s in ax.spines.values():
                    s.set_color(GRID)
                ax.set_xticks([]); ax.set_yticks([])
                sub = (f"HR@10 {met[0]:.3f}  ·  NDCG@10 {met[1]:.3f}" if met
                       else "no model trained")
                ax.set_title(f"{title}\n{sub}", fontsize=11.5, color=INK, pad=8)
            cb = fig.colorbar(sc, ax=axes, fraction=0.011, pad=0.01)
            cb.set_label("popularity vs median item  (log₂ ratio)", fontsize=9, color=INK2)
            cb.ax.tick_params(colors=INK2, labelsize=8)
            cb.outline.set_edgecolor(GRID)
            if mode == "genres":
                handles = [plt.Line2D([], [], marker="s", ls="", markerfacecolor=gcolors[g],
                                      markeredgecolor="none", markersize=10, label=g)
                           for g in gorder]
                fig.legend(handles=handles, loc="lower center", ncol=len(gorder),
                           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
                head = "genre territories"
            else:
                head = "clusters found in each geometry's own space"
            fig.suptitle(f"MovieLens-1M — {N}×{M}, seed {SEED}: {head}",
                         fontsize=13.5, color=INK, y=0.99)
            out = os.path.join(OUTDIR, f"manifold4_ml1m_{N}x{M}_{mode}.png")
            fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print(f"[ok] {N}x{M} in {time.time()-t0:.0f}s", flush=True)

    json.dump(report, open(os.path.join(OUTDIR, "clusters32_genre_ami.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
