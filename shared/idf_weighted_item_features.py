"""IDF-weighted item features for GI-SASRec's initial Isomap graph.

Problem
-------
GI-SASRec builds its initial item geometry from the item x user matrix of mean ratings (values 1..5).
Because those vectors are sparse and integer-valued, item-item distances take only a few thousand distinct
values (office 1000x1200: ~1.5k distinct values among 720k pairs). The k-NN graph then becomes symmetric:
many items end up with bit-identical geodesic rows ("geodesic twins"), the centred Isomap kernel has exact
eigenvalue ties, and the backward pass of torch.linalg.eigh returns NaN. The NaN guard zeroes the whole outer
gradient, so the geometry never trains (moved 0.2% in 300 epochs on office 1000x1200).

Fix
---
Weight each user column by its smoothed inverse document frequency before computing distances:

    idf[u] = log((1 + n_items) / (1 + n_items_rated_by_u)) + 1        (sklearn's smooth_idf)
    feature[i, u] = mean_rating[i, u] * idf[u]

Users who rated few items get more weight. Distances become effectively continuous (office 1000x1200:
672k distinct values), twins and exact eigen-ties disappear (120 -> 0), and the geometry trains.
Nothing else in the pipeline changes.

Result (office 1000x1200, same 858-user sample, latent 64):
    GI-SASRec + IDF   test HR@10 0.383   (guard fired on 3% of outer epochs, geometry moved 78%)
    GI-SASRec raw     test HR@10 0.312   (guard on 100% of epochs, geometry moved 0.2%)
    classic SASRec    test HR@10 0.319 +- 0.002 (3 seeds)

Usage in sasrec_experiment_pipeline.py (one-line change, where the feature matrix is built):

    movie_user_mat = build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True)
    movie_user_mat = idf_item_features(movie_user_mat)          # <- add this line
    features_t = torch.tensor(movie_user_mat, dtype=torch.float32, device=device)
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import numpy as np


def idf_item_features(item_user_mat):
    """item_user_mat: (n_items, n_users) mean ratings, 0 = not rated. Returns float32 IDF-weighted features."""
    mat = np.asarray(item_user_mat, dtype=np.float64)
    n_items = mat.shape[0]
    items_per_user = (mat > 0).sum(axis=0)
    idf = np.log((1.0 + n_items) / (1.0 + items_per_user)) + 1.0
    return (mat * idf[None, :]).astype(np.float32)


def exact_eigen_ties(features, latent_len=64, n_neighbors=10, device="cuda"):
    """Diagnostic: number of exactly tied float32 eigenvalues of the initial Isomap kernel.

    Uses the project's own IsomapNN (needs the repo root on sys.path), exactly as GI-SASRec builds the
    initial geometry: Euclidean distances normalised to max 1 -> k-NN geodesics -> centred kernel -> eigh.
    Any value > 0 means eigh's backward returns NaN on the first outer step.
    (The tie pattern depends on the exact float32 geodesics, so a re-implementation, e.g. with scipy,
    does not reproduce the counts.)
    """
    import torch
    from collections import Counter
    from Adam.Isomap import IsomapNN

    f = torch.tensor(features, dtype=torch.float32, device=device)
    D = torch.cdist(f, f)
    D = D / D.max()
    captured = {}
    _eigh = torch.linalg.eigh

    def eigh_hook(K, *args, **kwargs):
        w, V = _eigh(K, *args, **kwargs)
        captured["w"] = w.detach()
        return w, V

    torch.linalg.eigh = eigh_hook
    try:
        with torch.no_grad():
            IsomapNN(D, n_components=latent_len, n_neighbors=n_neighbors, eigval_choice="MDS").to(device)()
    finally:
        torch.linalg.eigh = _eigh
    return sum(c for c in Counter(captured["w"].cpu().tolist()).values() if c > 1)
