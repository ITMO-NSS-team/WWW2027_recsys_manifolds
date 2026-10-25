"""Four-panel item clouds: how the geometry changes from raw data to fitted manifold.

Panels, left to right:
  1. Data geometry   - Euclidean distances between item rating profiles
                       (D_input_init). No model has touched this; it is where
                       the gradient-Isomap search starts. No metrics: nothing
                       was trained.
  2. Euclidean       - the item embedding table classic SASRec learned.
  3. Fixed Poincaré  - hyperbolic MDS at the data-derived curvature, frozen.
  4. Learned (ours)  - the manifold fitted to the recommendation objective.

Every caption's metrics come from the SAME artefact as the cloud above it
(each npz/json stores the score of the model whose geometry it holds), never
from the results table. An earlier version of this figure captioned panel 2
with table values while plotting a geometry trained at a different maxlen and
dropout - on Grocery that was 0.410 printed over a cloud that scored 0.354.

Projection is t-SNE on each geometry's own distance matrix, identical settings
across panels. Positions are comparable within a panel only. Edges are omitted:
the graphs are complete.

Colour is log2(popularity / median popularity) on a diverging red-blue ramp -
blue rarer than the median item, red more frequent, neutral at the median.
Popularity is a magnitude and would normally take a sequential ramp; giving the
ramp a real midpoint is what makes a diverging one legitimate here.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.manifold import TSNE

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)
from data_preparation_interactions import (  # noqa: E402
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
)

SHARE = os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external",
         "sasrec_main")
LOC = _paths.SASREC_RUNS_ROOT
OUTDIR = os.path.join(_paths.FIGURES_OUT_DIR, "manifold_panels4")
DATA_ROOT = _paths.DATA_DIR
INK, INK2, GRID = "#0b0b0b", "#52514e", "#deddd8"
SEED = 0
SCALES = [(300, 800), (1000, 1200), (3000, 2500)]
DATASETS = [
    ("Arts & Crafts", "arts", "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing"),
    ("Grocery", "grocery", "amazon_grocery_gourmet_food", "Grocery_and_Gourmet_Food"),
    ("Office Products", "office", "amazon_office_products", "Office_Products"),
    ("Amazon Beauty", "beauty", "amazon_beauty", "Beauty_and_Personal_Care"),
    ("MovieLens-1M", "ml1m", "ml-1m", None),
]


def popularity(dirname, category, N, M):
    if category is None:
        df = pd.read_csv(os.path.join(DATA_ROOT, dirname, "ratings.dat"), sep="::",
                         engine="python", header=None,
                         names=["userId", "movieId", "rating", "timestamp"])
    else:
        df = pd.read_csv(os.path.join(DATA_ROOT, dirname, category + ".csv"))
        df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
        df = df[["userId", "movieId", "rating", "timestamp"]]
    dfm, _ = prepare_sequences(df)
    _, u2s, _nu, ni = subsample_users_items(dfm, max_users=N, max_movies=M, min_seq_len=5)
    tr, _va, _te = train_val_test_split_next_item(u2s, min_len=3)
    pop = np.zeros(ni)
    for (_u, m, _r) in tr:
        pop[int(m)] += 1
    return pop


def ours_folder(tag, N, M):
    for root, pat in ((SHARE, f"sasrec_{tag}_lat32_{N}x{M}_seed{SEED}"),
                      (LOC, f"sasrec_{tag}_latent32_{N}x{M}_seed{SEED}"),
                      (LOC, f"sasrec_amazon_beauty_latent32_{N}x{M}_seed{SEED}")):
        f = os.path.join(root, "logs_sasrec_isomap", pat)
        if os.path.isdir(f):
            return f
    return None


def ours_metrics(tag, N, M):
    for root, pats in ((SHARE, [f"results_sasrec_{tag}_lat32_{N}x{M}_seed{SEED}.json"]),
                       (LOC, [f"results_sasrec_{tag}_latent32_{N}x{M}_seed{SEED}.json",
                              f"results_sasrec_amazon_beauty_latent32_{N}x{M}_seed{SEED}.json"])):
        for pat in pats:
            p = os.path.join(root, pat)
            if os.path.exists(p):
                d = json.load(open(p, encoding="utf-8")).get("gi_sasrec", {})
                if d.get("test_hr") is not None:
                    return d["test_hr"], d["test_ndcg"]
    return None


def collect(tag, N, M):
    """(label, distance-matrix path, caption, metrics or None) per panel."""
    folder = ours_folder(tag, N, M)
    init = os.path.join(folder, "D_input_init.npy") if folder else None
    snaps = glob.glob(os.path.join(folder, "matrices_epoch*.npz")) if folder else []
    last = max(snaps, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if snaps else None

    eu = os.path.join(LOC, "euclidean_geometry_aligned",
                      f"euclidean_geometry_{tag}_{N}x{M}_seed{SEED}.npz")
    eu_m = None
    if os.path.exists(eu):
        d = np.load(eu)
        eu_m = (float(d["test_hr"]), float(d["test_ndcg"]))
    else:
        eu = None

    po = os.path.join(LOC, "poincare_fixed", f"geometry_{tag}_{N}x{M}_seed{SEED}.npz")
    po_res = os.path.join(LOC, "poincare_fixed", f"results_{tag}_{N}x{M}_seed{SEED}.json")
    po_m = None
    if os.path.exists(po_res):
        d = json.load(open(po_res, encoding="utf-8"))
        po_m = (d["test_hr"], d["test_ndcg"])
    po = po if os.path.exists(po) else None

    return [
        ("Data geometry", init, "item rating profiles, before any model", None),
        ("Euclidean", eu, "classic SASRec embedding table", eu_m),
        ("Fixed Poincaré", po, "hyperbolic MDS, derived c", po_m),
        ("Learned (ours)", last, "gradient Isomap, fitted to the task",
         ours_metrics(tag, N, M)),
    ]


def load_D(path):
    if path.endswith(".npy"):
        D = np.load(path).astype(np.float64)
    else:
        d = np.load(path)
        D = np.asarray(d["D"] if "D" in d else d["D_latent"], dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, D.T)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    made, skipped = [], []
    for ds_name, tag, dirname, category in DATASETS:
        for (N, M) in SCALES:
            panels = collect(tag, N, M)
            miss = [p[0] for p in panels if not p[1]]
            if miss:
                skipped.append(f"{ds_name} {N}x{M}: missing {', '.join(miss)}")
                continue
            t0 = time.time()
            pop = popularity(dirname, category, N, M)
            fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.9))
            sc = None
            for ax, (title, path, cap, met) in zip(axes, panels):
                D = load_D(path)
                n = min(len(pop), D.shape[0])
                D, c = D[:n, :n], pop[:n]
                med = max(np.median(c), 1.0)
                cval = np.log2(np.maximum(c, 0.5) / med)
                lim = float(np.percentile(np.abs(cval), 98)) or 1.0
                xy = TSNE(n_components=2, metric="precomputed", init="random",
                          perplexity=30, random_state=0, max_iter=1000).fit_transform(D)
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=cval, s=9, cmap="RdBu_r",
                                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                                linewidth=0, alpha=0.88)
                ax.set_aspect("equal", adjustable="box")
                ax.set_facecolor("#fcfcfb")
                for s in ax.spines.values():
                    s.set_color(GRID)
                ax.set_xticks([]); ax.set_yticks([])
                sub = (f"HR@10 {met[0]:.3f}  ·  NDCG@10 {met[1]:.3f}" if met
                       else "no model trained")
                ax.set_title(f"{title}\n{sub}", fontsize=11.5, color=INK, pad=8)
                ax.text(0.5, -0.055, cap, transform=ax.transAxes, ha="center",
                        fontsize=8.5, color=INK2)
            cb = fig.colorbar(sc, ax=axes, fraction=0.011, pad=0.01)
            cb.set_label("popularity vs median item  (log₂ ratio)", fontsize=9, color=INK2)
            cb.ax.tick_params(colors=INK2, labelsize=8)
            cb.outline.set_edgecolor(GRID)
            fig.suptitle(f"{ds_name} — {N}×{M}, seed {SEED}: the same {n} items, "
                         f"four geometries", fontsize=13.5, color=INK, y=0.99)
            fig.text(0.5, -0.015, "t-SNE on each geometry's own distance matrix, identical "
                     "settings. Comparable within a panel, not between. Metrics are those of "
                     "the model that produced that panel's geometry.",
                     ha="center", fontsize=8.5, color=INK2)
            out = os.path.join(OUTDIR, f"manifold4_{tag}_{N}x{M}.png")
            fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            made.append(os.path.basename(out))
            print(f"[ok] {ds_name} {N}x{M} ({n}) in {time.time()-t0:.0f}s", flush=True)

    print(f"\n=== {len(made)} figures in {OUTDIR} ===")
    for s in skipped:
        print(f"   skipped - {s}")


if __name__ == "__main__":
    main()
