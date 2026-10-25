"""GI-SASRec on Amazon Beauty, 6000 users x 3000 items, seed 0.

GI-SASRec at the default latent_len=64, without inner warm start and
without regularisation (dropout 0, weight decay 0). This is the
configuration the adopted "ours-32" arm is compared against.

This arm: 300 outer epochs, hidden_units=64.

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

if __name__ == "__main__":
    sasrec_exp.main(
        max_users=6000,
        max_movies=3000,
        min_seq_len=5,
        top_k=10,
        gradisomap_epochs=300,
        cf_epochs=100,
        final_cf_epochs=50,
        lr_isomap=1e-3,
        n_run="sasrec_amazon_beauty_noreg_nowarm_6000x3000_seed0",
        seed=0,
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
        matrices_save_every=10,
    )
