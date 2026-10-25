"""GI-NCF on MovieLens-1M: any scale x seed.

Unlike the Amazon Beauty GI-NCF arms (see ncf/arms/), the MovieLens-1M GI-NCF
grid was produced with the vectorised pipeline in this package, driven from the
command line. This script pins the configurations that were actually run, so
the grid can be reproduced without reconstructing the flags by hand. Every
argument list below is transcribed from the `argv` field recorded in the run
log of a finished run.

Configurations
--------------
``paper`` (the default; run tree ``patience_300``)
    outer lr 0.001, frozen item projection, selection by validation HR@10,
    300 outer epochs, 200 inner CF epochs, patience 30 everywhere.
    This matches the convention used by every other arm in the paper.

``paper-no-resample`` (run tree ``patience_300_no_resample``)
    identical, except the training negatives are sampled once at the start
    instead of being redrawn every epoch.

Examples
--------
    python arm_gincf_ml1m_any_scale_seed.py --scale 1000x1200 --seed 0
    python arm_gincf_ml1m_any_scale_seed.py --scale 300x800 --seed 1 --dry-run
    python arm_gincf_ml1m_any_scale_seed.py --scale 3000x2500 --seed 2 \
        --config paper-no-resample
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import argparse
import os
import subprocess
import sys

PIPELINE = os.path.join(_paths.NCF_FAST_DIR,
                        "ncf_and_nonparametric_experiment_pipeline.py")

SCALES = {"300x800": (300, 800), "1000x1200": (1000, 1200),
          "3000x2500": (3000, 2500)}

# Transcribed verbatim from the recorded argv of the finished runs; only
# --max-users / --max-movies / --seed / --n-run vary across cells.
COMMON_PAPER = [
    "--dataset", "ml-1m", "--trainer", "fast", "--run-gincf",
    "--use-init-assumption", "--min-seq-len", "2", "--num-ng", "2",
    "--use-ratings", "--neg-exclude", "train", "--freeze-item-projection",
    "--latent-len", "64", "--factor-num", "16", "--num-layers", "3",
    "--n-neighbors", "10", "--lr-isomap", "0.001", "--lr-ncf", "0.001",
    "--batch-size", "2048", "--select-by", "hr",
    "--cf-epochs", "200", "--inner-patience", "30",
    "--final-cf-epochs", "200", "--final-patience", "30",
    "--epochs-pure", "200", "--ncf-patience", "30",
    "--gradisomap-epochs", "300", "--save-matrices",
]

CONFIGS = {
    "paper": COMMON_PAPER,
    "paper-no-resample": (COMMON_PAPER[:3] + ["--no-resample-negatives"] +
                          COMMON_PAPER[3:]),
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", required=True, choices=sorted(SCALES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default="paper", choices=sorted(CONFIGS))
    p.add_argument("--dry-run", action="store_true",
                   help="print the command instead of running it")
    args = p.parse_args()

    users, items = SCALES[args.scale]
    n_run = (f"gincf_ml1m_{args.config.replace('-', '_')}_"
             f"{args.scale}_seed{args.seed}")
    cmd = ([sys.executable, PIPELINE] + CONFIGS[args.config] +
           ["--max-users", str(users), "--max-movies", str(items),
            "--seed", str(args.seed), "--n-run", n_run])

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
