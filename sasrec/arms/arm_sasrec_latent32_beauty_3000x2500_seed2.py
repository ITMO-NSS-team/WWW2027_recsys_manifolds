"""GI-SASRec on Amazon Beauty, 3000 users x 2500 items, seed 2.

The adopted "ours-32" configuration: the Isomap latent length is cut to 32
while hidden_units stays 64. The cut is at a wider gap in the Isomap
eigenvalue spectrum than the default 64, which keeps the backward pass of the
eigendecomposition better conditioned; 32 is also the dimension the
HypSASRec authors report as empirically best.

This arm: 300 outer epochs, latent_len=32, hidden_units=64.

The hyperparameters below are the record of what was run; the pipeline
itself lives one directory up.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sasrec_experiment_pipeline as sasrec_exp

SEED = 2

if __name__ == "__main__":
    sasrec_exp.main(
        max_users=3000,
        max_movies=2500,
        min_seq_len=5,
        top_k=10,
        gradisomap_epochs=300,
        cf_epochs=100,
        final_cf_epochs=50,
        lr_isomap=1e-3,
        n_run=f"sasrec_amazon_beauty_latent32_3000x2500_seed{SEED}",
        seed=SEED,
        dataset_dir_name="amazon_beauty",
        dataset_type="amazon",
        amazon_category="Beauty_and_Personal_Care",
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
