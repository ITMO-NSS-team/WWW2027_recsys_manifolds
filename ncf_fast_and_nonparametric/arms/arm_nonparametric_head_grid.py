"""Non-parametric heads (ItemKNN, bary_cos) on the gradient-Isomap geometry.

These heads have ``trainable_head_params = 0``: there is no model to train on
top of the geometry, so the outer loop has to move the geometry itself for the
head to gain anything. That is the point of the arm — it isolates how much
deformation each kind of head demands.

Every cell of the grid that was actually run is recorded in
``results/tables/nonparametric_grid_configs.json`` (dataset x head x
hyperparameter combo x scale, with the exact command-line flags transcribed
from the finished runs' logs). This script replays any subset of it.

Why the grid is stored as data rather than hard-coded: ``--np-bias`` is not a
free choice. It is the offset that places the decision threshold midway
between the positive and negative score distributions, which the trainer
prints at initialisation, so it differs per (head, combo, scale) while being
identical across seeds. Keeping the measured values alongside the flags is the
only way to replay a cell exactly.

Examples
--------
    # what is in the grid
    python arm_nonparametric_head_grid.py --list

    # one cell, all three seeds
    python arm_nonparametric_head_grid.py --dataset ml-1m --head item_knn \
        --scale 1000x1200 --combo nn_10_lr_1e-3/knn_k_20_temp_0.5

    # everything for one head on one dataset, printed but not run
    python arm_nonparametric_head_grid.py --dataset ml-1m --head bary_cos \
        --dry-run
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import sys

PIPELINE = os.path.join(_paths.NCF_FAST_DIR,
                        "ncf_and_nonparametric_experiment_pipeline.py")
GRID = os.path.join(_paths.TABLES_DIR, "nonparametric_grid_configs.json")

SCALES = {"300x800": (300, 800), "1000x1200": (1000, 1200),
          "3000x2500": (3000, 2500)}
# the dataset key in the grid file -> the pipeline's --dataset value
DATASET_FLAG = {"ml-1m": "ml-1m", "amazon-beauty": "amazon"}


def load_grid():
    with open(GRID, encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="print the grid and exit")
    p.add_argument("--dataset", choices=sorted(DATASET_FLAG))
    p.add_argument("--head", choices=["bary_cos", "item_knn"])
    p.add_argument("--combo", help="hyperparameter combo, as printed by --list")
    p.add_argument("--scale", choices=sorted(SCALES))
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated seeds (default 0,1,2)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands instead of running them")
    args = p.parse_args()

    grid = load_grid()
    if args.list:
        for r in grid:
            print(f"{r['dataset']:14s} {r['head']:9s} {r['scale']:10s} "
                  f"{r['combo']}")
        print(f"\n{len(grid)} cells "
              f"(x the seeds given by --seeds, default three)")
        return

    sel = [r for r in grid
           if (args.dataset is None or r["dataset"] == args.dataset)
           and (args.head is None or r["head"] == args.head)
           and (args.combo is None or r["combo"] == args.combo)
           and (args.scale is None or r["scale"] == args.scale)]
    if not sel:
        raise SystemExit("no grid cell matches those filters; try --list")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"{len(sel)} cell(s) x {len(seeds)} seed(s) = "
          f"{len(sel) * len(seeds)} runs\n", flush=True)

    failures = []
    for r in sel:
        users, items = SCALES[r["scale"]]
        for seed in seeds:
            n_run = (f"np_{r['dataset']}_{r['head']}_"
                     f"{r['combo'].replace('/', '_')}_{r['scale']}_seed{seed}")
            cmd = ([sys.executable, PIPELINE, "--dataset",
                    DATASET_FLAG[r["dataset"]]] + r["flags"] +
                   ["--max-users", str(users), "--max-movies", str(items),
                    "--seed", str(seed), "--n-run", n_run])
            print(" ".join(cmd), flush=True)
            if args.dry_run:
                continue
            rc = subprocess.call(cmd)
            if rc != 0:
                failures.append((n_run, rc))

    if failures:
        print("\nfailed runs:", flush=True)
        for n, rc in failures:
            print(f"  {n} (exit {rc})", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
