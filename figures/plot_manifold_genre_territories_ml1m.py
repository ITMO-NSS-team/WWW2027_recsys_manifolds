"""MovieLens-1M four-panel clouds with the backdrop tinted by genre territory.

Question this answers: do the regions the learned geometry carves out correspond
to anything semantic, or are they an artefact of the optimisation?

Backdrop construction: each item is assigned a primary genre (its rarest genre
in the subsample, so that a film tagged Drama|Film-Noir counts as Film-Noir -
otherwise Drama and Comedy swallow everything). A kernel density field is built
per genre over the 2-D coordinates, and each grid cell is tinted by whichever
genre dominates it, at low alpha. The result is a territory map rather than a
per-point colouring, which is what makes the association visible at a glance.

ML-1M has 18 genres, far past the 8 a categorical palette can keep separable,
so the seven most frequent genres in each subsample get the validated hue slots
and everything else folds into a neutral "Other" - the documented way to handle
an over-long category list.

Points are drawn in neutral ink here, NOT the popularity ramp: the backdrop is
already carrying a colour encoding, and two colour channels in one panel would
collide.

Whether the association is real is not left to the eye - adjusted mutual
information between HDBSCAN regions and primary genre is printed to stdout for
every panel. AMI is 0 for independent labellings and 1 for identical ones, and
it is adjusted for chance, so a count of 20 regions cannot inflate it.
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
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_manifold_projection_panels import SCALES, SEED, INK, INK2, GRID, collect, load_D  # noqa: E402

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)
from data_preparation_interactions import prepare_sequences, subsample_users_items  # noqa: E402

ML1M = os.path.join(_paths.DATA_DIR, "ml-1m")
OUTDIR = os.path.join(_paths.FIGURES_OUT_DIR, "manifold_panels4_genres")
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER = "#b9b9b1"


def genre_of_items(N, M):
    """Primary genre per final item index, after both re-indexings."""
    ratings = pd.read_csv(os.path.join(ML1M, "ratings.dat"), sep="::", engine="python",
                          header=None, names=["userId", "movieId", "rating", "timestamp"])
    dfm, _ = prepare_sequences(ratings)
    df_sub, _u2s, _nu, ni = subsample_users_items(dfm, max_users=N, max_movies=M,
                                                  min_seq_len=5)
    idx2movie = (df_sub.drop_duplicates("movie_idx")
                 .set_index("movie_idx")["movieId"].to_dict())

    movies = pd.read_csv(os.path.join(ML1M, "movies.dat"), sep="::", engine="python",
                         header=None, names=["movieId", "title", "genres"],
                         encoding="latin-1")
    gmap = dict(zip(movies["movieId"], movies["genres"]))

    raw = {i: str(gmap.get(idx2movie.get(i, -1), "")).split("|") for i in range(ni)}
    freq = Counter(g for gs in raw.values() for g in gs if g)
    # rarest-genre-wins: Drama and Comedy otherwise absorb most of the catalogue
    primary = {}
    for i, gs in raw.items():
        gs = [g for g in gs if g]
        primary[i] = min(gs, key=lambda g: freq[g]) if gs else "Other"
    top = [g for g, _ in Counter(primary.values()).most_common(len(HUES))]
    labels = np.array([primary[i] if primary[i] in top else "Other" for i in range(ni)])
    order = top + (["Other"] if (labels == "Other").any() else [])
    return labels, order, ni


def genre_backdrop(ax, xy, labels, order, colors, bins=170, sigma=4.0, q=55):
    x, y = xy[:, 0], xy[:, 1]
    xr, yr = x.max() - x.min(), y.max() - y.min()
    pad = 0.07
    ext = [x.min() - pad * xr, x.max() + pad * xr, y.min() - pad * yr, y.max() + pad * yr]
    rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
    fields, tot = [], None
    for g in order:
        sel = labels == g
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
        rgba[m, 3] = 0.30
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear", zorder=1,
              aspect="auto")
    return ext


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    for (N, M) in SCALES:
        panels = collect("ml1m", N, M)
        if any(not p[1] for p in panels):
            print(f"skip {N}x{M}", flush=True)
            continue
        t0 = time.time()
        labels, order, ni = genre_of_items(N, M)
        colors = {g: (HUES[i] if g != "Other" else OTHER) for i, g in enumerate(order)}
        fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.9))
        for ax, (title, path, cap, met) in zip(axes, panels):
            D = load_D(path)
            n = min(len(labels), D.shape[0])
            D, lab = D[:n, :n], labels[:n]
            xy = TSNE(n_components=2, metric="precomputed", init="random",
                      perplexity=30, random_state=0, max_iter=1000).fit_transform(D)
            ext = genre_backdrop(ax, xy, lab, order, colors)
            ax.scatter(xy[:, 0], xy[:, 1], s=7, c="#3a3a38", linewidth=0, alpha=0.75,
                       zorder=3)
            reg = HDBSCAN(min_cluster_size=max(8, n // 120), copy=True).fit_predict(xy)
            keep = reg >= 0
            ami = (adjusted_mutual_info_score(lab[keep], reg[keep]) if keep.sum() > 10
                   else float("nan"))
            report.append({"scale": f"{N}x{M}", "panel": title, "AMI": round(float(ami), 4),
                           "regions": int(len(set(reg[keep])))})
            print(f"  {N}x{M} {title:16s} AMI(genre, region) = {ami:.3f}  "
                  f"regions {len(set(reg[keep]))}", flush=True)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("#fcfcfb")
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            sub = (f"HR@10 {met[0]:.3f}  ·  NDCG@10 {met[1]:.3f}" if met
                   else "no model trained")
            ax.set_title(f"{title}\n{sub}", fontsize=11.5, color=INK, pad=8)
        handles = [plt.Line2D([], [], marker="s", ls="", markerfacecolor=colors[g],
                              markeredgecolor="none", markersize=10, label=g) for g in order]
        fig.legend(handles=handles, loc="lower center", ncol=len(order), frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"MovieLens-1M — {N}×{M}, seed {SEED}: genre territories under "
                     f"four geometries", fontsize=13.5, color=INK, y=0.99)
        out = os.path.join(OUTDIR, f"manifold4genre_ml1m_{N}x{M}.png")
        fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[ok] {N}x{M} in {time.time()-t0:.0f}s -> {os.path.basename(out)}", flush=True)
    json.dump(report, open(os.path.join(OUTDIR, "genre_region_ami.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
