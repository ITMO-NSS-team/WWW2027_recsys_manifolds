"""Parameterised GI-SASRec ("ours-32") arm: any dataset x scale x seed.

The per-file drivers next to this one cover MovieLens-1M and Amazon Beauty,
the two datasets trained on the machine this repository was assembled from.
The Arts / Grocery / Office runs of the same arm were executed on a second
machine with the identical configuration; this script reproduces any cell of
the grid, including those three, from one entry point.

The configuration below is copied verbatim from
arm_sasrec_latent32_beauty_3000x2500_seed0.py - the adopted "ours-32" setting:
latent_len=32 with hidden_units=64, maxlen=30, no dropout, no weight decay,
no inner warm start, frozen item projection, outer lr 1e-3, 300 outer epochs,
checkpoint selection by validation HR@10.

Examples
--------
    python arm_sasrec_latent32_any_dataset_scale_seed.py \
        --dataset grocery --scale 3000x2500 --seed 1

    for s in 0 1 2; do
      python arm_sasrec_latent32_any_dataset_scale_seed.py \
          --dataset office --scale 1000x1200 --seed $s
    done
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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sasrec_experiment_pipeline as sasrec_exp  # noqa: E402

DATASETS = {
    "ml1m": dict(dataset_dir_name="ml-1m", dataset_type="movielens",
                 amazon_category=None),
    "beauty": dict(dataset_dir_name="amazon_beauty", dataset_type="amazon",
                   amazon_category="Beauty_and_Personal_Care"),
    "arts": dict(dataset_dir_name="amazon_arts_crafts_sewing",
                 dataset_type="amazon",
                 amazon_category="Arts_Crafts_and_Sewing"),
    "grocery": dict(dataset_dir_name="amazon_grocery_gourmet_food",
                    dataset_type="amazon",
                    amazon_category="Grocery_and_Gourmet_Food"),
    "office": dict(dataset_dir_name="amazon_office_products",
                   dataset_type="amazon",
                   amazon_category="Office_Products"),
}

SCALES = {"300x800": (300, 800), "1000x1200": (1000, 1200),
          "3000x2500": (3000, 2500), "6000x3000": (6000, 3000)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--scale", required=True, choices=sorted(SCALES))
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    max_users, max_movies = SCALES[args.scale]
    n_run = f"sasrec_{args.dataset}_latent32_{args.scale}_seed{args.seed}"
    print(f"\n=== {n_run} ===\n", flush=True)

    sasrec_exp.main(
        max_users=max_users,
        max_movies=max_movies,
        min_seq_len=5,
        top_k=10,
        gradisomap_epochs=300,
        cf_epochs=100,
        final_cf_epochs=50,
        lr_isomap=1e-3,
        n_run=n_run,
        seed=args.seed,
        dataset_dir_name=ds["dataset_dir_name"],
        dataset_type=ds["dataset_type"],
        amazon_category=ds["amazon_category"] or "",
        select_by="hr",
        final_patience=30,
        inner_patience=30,
        warm_start_inner=False,
        freeze_item_projection=True,
        maxlen=30,
        hidden_units=64,
        num_blocks=2,
        num_heads=1,
        sasrec_dropout=0.0,
        weight_decay=0.0,
        latent_len=32,
    )


if __name__ == "__main__":
    main()
