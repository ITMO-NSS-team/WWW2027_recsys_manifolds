"""What distinguishes the GI-NCF run trees found in EXTERNAL_RUNS_DIR/GINCF.

Several trees exist there and only one of them, `patience_300`, is the paper
configuration (lr_isomap=0.001, frozen item projection, 300 outer epochs,
select_by=hr). The others are hyperparameter searches and are not analysed
anywhere in this repository.

This script prints, per tree, the run count, history length, statuses, scales,
and exactly which config fields differ between trees - the check that
establishes which tree a given set of numbers came from before anything is
aggregated.
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

NCF = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GINCF")
FIELDS = ["dataset", "amazon_category", "max_users", "max_movies", "seed",
          "gradisomap_epochs", "outer_patience", "cf_epochs", "inner_patience",
          "lr_isomap", "n_neighbors", "latent_len", "factor_num", "num_layers",
          "freeze_item_projection", "use_item_projection", "dropout",
          "warm_start_inner", "select_by", "dist_type"]


def main():
    trees = {}
    for lg in sorted(glob.glob(os.path.join(NCF, "*", "*", "run_log.json"))):
        tree = os.path.relpath(os.path.dirname(lg), NCF).split(os.sep)[0]
        with open(lg) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        gi = d.get("gi", {})
        n_ep = len(gi.get("history", {}).get("epoch", []))
        trees.setdefault(tree, []).append((cfg, n_ep, d.get("status"),
                                           gi.get("test", {})))

    for tree, runs in trees.items():
        print("=" * 78)
        print(f"{tree}   runs: {len(runs)}")
        print("=" * 78)
        eps = sorted({n for _c, n, _s, _t in runs})
        print(f"  history length: {eps}")
        print(f"  statuses: {sorted({s for _c, _n, s, _t in runs})}")
        scales = sorted({f'{c.get("max_users")}x{c.get("max_movies")}'
                         for c, _n, _s, _t in runs})
        print(f"  scales: {scales}")
        hrs = [t.get("hr") for _c, _n, _s, t in runs if t and t.get("hr")]
        if hrs:
            print(f"  test HR: {min(hrs):.4f}..{max(hrs):.4f}")
        print("  config fields that differ:")
        for f in FIELDS:
            vals = sorted({str(c.get(f)) for c, _n, _s, _t in runs})
            print(f"      {f:26s} {vals if len(vals) <= 4 else vals[:4] + ['...']}")
        print()

    print("=" * 78)
    print("config comparison across run trees (fields where they differ)")
    print("=" * 78)
    names = list(trees)
    for f in FIELDS:
        per = {t: sorted({str(c.get(f)) for c, _n, _s, _t in runs})
               for t, runs in trees.items()}
        if len({tuple(v) for v in per.values()}) > 1:
            print(f"  {f}:")
            for t in names:
                print(f"      {t:34s} {per[t]}")


if __name__ == "__main__":
    main()
