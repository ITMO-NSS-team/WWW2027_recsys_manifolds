"""
H1 (cycles) for the large-scale GINCF runs, where geometry_topology_diagnostics.py
skipped it: its max_simplices=300_000 guard trips at 2500+ items, so
3000x2500 and 6000x3000 have no cycle data at all while 300x800 has 94%
coverage. That leaves the scale story with cycle evidence only where the
method works and none where it fails.

Brute-forcing a full Vietoris-Rips H1 on 2500-3000 points is what the guard
exists to prevent. Instead we use the same device already used for
delta-hyperbolicity: estimate on random item subsets. n_sub items are drawn
per repeat, H1 is computed on that submatrix, and we report mean/std over
repeats. This bounds cost and carries its own error bars, at the price of
measuring cycles of the subsampled complex rather than the full one -
comparable across epochs and runs (same n_sub, same seeds), not comparable
to the full-complex numbers at 300x800.

Compares the first and the last saved epoch of each run, on D_geodesic and
D_latent. Read-only; writes one JSON next to itself.
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
import time

import numpy as np

MANUL = _paths.REPO_ROOT
LOGS = _paths.NCF_RUNS_DIR
HERE = os.path.dirname(os.path.abspath(__file__))

import gudhi  # noqa: E402

N_SUB = 900          # keeps the Rips complex comfortably inside the guard's budget
N_REPEATS = 3
SEED = 7
RUN_GLOB = [
    "amazon_beauty_frozenproj_scaletest_3000x2500_0.001",
    "amazon_beauty_frozenproj_scaletest_6000x3000_seed0_0.001",
    "amazon_beauty_frozenproj_scaletest_6000x3000_seed1_0.001",
    "amazon_beauty_frozenproj_scaletest_6000x3000_seed2_0.001",
]


def h1_on(D, max_edge=None):
    """H1 count and max persistence of the Vietoris-Rips complex of D."""
    if max_edge is None:
        max_edge = float(np.percentile(D[np.triu_indices_from(D, k=1)], 50))
    rips = gudhi.RipsComplex(distance_matrix=D.astype(np.float64), max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2)
    iv = st.persistence_intervals_in_dimension(1)
    if len(iv) == 0:
        return 0, 0.0
    finite = iv[np.isfinite(iv[:, 1])]
    if len(finite) == 0:
        return int(len(iv)), float("nan")
    return int(len(iv)), float(np.max(finite[:, 1] - finite[:, 0]))


def epoch_num(p):
    return int(re.search(r"epoch(\d+)", p).group(1))


def main():
    results = []
    for run in RUN_GLOB:
        folder = os.path.join(LOGS, run)
        snaps = sorted(glob.glob(os.path.join(folder, "matrices_epoch*.npz")), key=epoch_num)
        if len(snaps) < 2:
            print(f"[skip] {run}: {len(snaps)} snapshots", flush=True)
            continue
        print(f"\n=== {run}: {len(snaps)} snapshots, epochs "
              f"{epoch_num(snaps[0])}..{epoch_num(snaps[-1])} ===", flush=True)
        for which, path in (("first", snaps[0]), ("last", snaps[-1])):
            d = np.load(path)
            n = d["D_geodesic"].shape[0]
            rng = np.random.default_rng(SEED)
            for space in ("D_geodesic", "D_latent"):
                D_full = d[space]
                counts, pers = [], []
                for rep in range(N_REPEATS):
                    idx = rng.choice(n, min(N_SUB, n), replace=False)
                    t0 = time.time()
                    c, p = h1_on(D_full[np.ix_(idx, idx)])
                    counts.append(c)
                    pers.append(p)
                    print(f"   {which:5s} {space:11s} rep{rep}: H1={c:5d} "
                          f"maxpers={p:.4f} ({time.time() - t0:.0f}s)", flush=True)
                results.append({
                    "run": run, "epoch": epoch_num(path), "which": which, "space": space,
                    "n_items": int(n), "n_sub": int(min(N_SUB, n)),
                    "h1_count_mean": float(np.mean(counts)), "h1_count_std": float(np.std(counts)),
                    "h1_maxpers_mean": float(np.nanmean(pers)), "h1_maxpers_std": float(np.nanstd(pers)),
                })
                with open(os.path.join(HERE, "h1_large_scales.json"), "w") as f:
                    json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['run'][-28:]:28s} ep.{r['epoch']:3d} {r['which']:5s} {r['space']:11s} "
              f"H1={r['h1_count_mean']:7.1f}±{r['h1_count_std']:5.1f} "
              f"maxpers={r['h1_maxpers_mean']:.4f}±{r['h1_maxpers_std']:.4f}")


if __name__ == "__main__":
    main()
