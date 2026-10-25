"""Figure: delta-hyperbolicity is a property of (data, representation rank).

The hyperbolic recommender literature reports "the delta-hyperbolicity of a
dataset" as if it were a property of the data. It is not: delta_rel is computed
on some representation of the interaction matrix, and the representation has a
capacity. This sweeps the SVD rank for all five datasets at a fixed subsample
and shows both delta_rel and the curvature derived from it through the authors'
formula, so the size of the effect is visible in the units a practitioner would
actually act on.

Two stacked panels rather than one with two y-axes: a dual-axis chart makes the
crossing point of the two curves an artefact of the scaling, and there is no
honest way to choose that scaling.

Computation matches `compute_hypsasrec_curvature.py` exactly - binary training
matrix, item representations from the rank-d truncated SVD, delta_rel = 2*delta
/diam over the full item set (<= 1500 items here, so no batching), Poincare
anchor at tol=1e-12. CPU only.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)
sys.path.insert(0, _paths.ANALYSIS_DIR)
from data_preparation_interactions import (  # noqa: E402
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
)
from gromov_delta_hyperbolicity import delta_hyperbolicity  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _paths.DATA_DIR
OUTDIR = _paths.FIGURES_OUT_DIR
CACHE = os.path.join(_paths.TABLES_DIR, "delta_rank_sweep.json")

N, M = 1000, 1200
RANKS = [8, 16, 32, 64, 128, 256, None]      # None = full rank
MODEL_RANK = 32
INK, INK2, GRID = "#0b0b0b", "#52514e", "#deddd8"
DATASETS = [
    ("MovieLens-1M", "ml-1m", None, "#2a78d6"),
    ("Amazon Beauty", "amazon_beauty", "Beauty_and_Personal_Care", "#eb6834"),
    ("Arts & Crafts", "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing", "#1baf7a"),
    ("Grocery", "amazon_grocery_gourmet_food", "Grocery_and_Gourmet_Food", "#eda100"),
    ("Office Products", "amazon_office_products", "Office_Products", "#e87ba4"),
]


def anchor(tol=1e-12):
    r = 1.0 - tol
    return np.log1p(np.sqrt(2)) / np.log1p(2 * r / (1 - r))


def load(dirname, category):
    if category is None:
        df = pd.read_csv(os.path.join(DATA, dirname, "ratings.dat"), sep="::",
                         engine="python", header=None,
                         names=["userId", "movieId", "rating", "timestamp"])
    else:
        df = pd.read_csv(os.path.join(DATA, dirname, category + ".csv"))
        df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})
        df = df[["userId", "movieId", "rating", "timestamp"]]
    return df


def sweep_one(dirname, category):
    dfm, _ = prepare_sequences(load(dirname, category))
    _, u2s, nu, ni = subsample_users_items(dfm, max_users=N, max_movies=M, min_seq_len=5)
    tr, _va, _te = train_val_test_split_next_item(u2s, min_len=3)
    mat = np.zeros((ni, nu), dtype=np.float32)
    for (u, m, _r) in tr:
        mat[int(m), int(u)] = 1.0
    U, S, _ = np.linalg.svd(mat, full_matrices=False)
    order = np.argsort(S)[::-1]
    out = []
    for rank in RANKS:
        k = len(S) if rank is None else min(rank, len(S))
        X = np.ascontiguousarray(U[:, order[:k]] * S[order[:k]]).astype(np.float64)
        cond = pdist(X)
        diam = float(cond.max())
        d = delta_hyperbolicity(squareform(cond), basepoint=0)
        dr = 2.0 * d / diam
        out.append({"rank": k, "is_full": rank is None, "delta_rel": float(dr),
                    "c": float((anchor() / dr) ** 2)})
        print(f"      rank {k:4d}: delta_rel={dr:.4f}  c={out[-1]['c']:.5f}", flush=True)
    return out, int(ni)


def compute():
    res = {}
    for name, dirname, category, _col in DATASETS:
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        rows, ni = sweep_one(dirname, category)
        res[name] = {"items": ni, "sweep": rows}
        print(f"    {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(CACHE, "w"), indent=1)
    return res


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw(res):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True)
    for ax in axes:
        style(ax)
        ax.axvline(MODEL_RANK, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
    axes[0].text(MODEL_RANK * 1.08, 0.425, "rank used\nby the models",
                 fontsize=8.5, color=INK2, va="top")
    for name, _d, _c, col in DATASETS:
        if name not in res:
            continue
        sw = res[name]["sweep"]
        x = [r["rank"] for r in sw]
        axes[0].plot(x, [r["delta_rel"] for r in sw], color=col, lw=2,
                     marker="o", ms=5, label=name, zorder=3)
        axes[1].plot(x, [r["c"] for r in sw], color=col, lw=2, marker="o", ms=5,
                     zorder=3)
    axes[0].set_ylabel(r"$\delta_{\mathrm{rel}}$ of the data", fontsize=10, color=INK)
    axes[1].set_ylabel("curvature $c$ derived from it", fontsize=10, color=INK)
    axes[1].set_xlabel("rank of the representation the measurement is taken on",
                       fontsize=10, color=INK)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([8, 16, 32, 64, 128, 256, 1200])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    axes[0].set_title("One dataset does not have one hyperbolicity",
                      fontsize=12.5, color=INK, pad=10, loc="left")
    fig.text(0.5, -0.01, f"All five datasets at the {N}×{M} subsample. Both quantities "
             "are measured exactly as the hyperbolic\nrecommender literature measures them; "
             "only the rank of the representation changes.",
             ha="center", fontsize=8.5, color=INK2)
    out = os.path.join(OUTDIR, "delta_rank_dependence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[Save] {out}")


def main():
    res = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else compute()
    draw(res)
    print("\n=== summary: factor by which c changes from rank 8 to full rank ===")
    for name in res:
        sw = res[name]["sweep"]
        print(f"  {name:16s} delta_rel {sw[0]['delta_rel']:.4f} -> {sw[-1]['delta_rel']:.4f}   "
              f"c {sw[0]['c']:.5f} -> {sw[-1]['c']:.5f}  ({sw[-1]['c']/sw[0]['c']:.1f}x)")


if __name__ == "__main__":
    main()
