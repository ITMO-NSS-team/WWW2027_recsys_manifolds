"""GI-SASRec on MovieLens-1M, 300 users x 800 items, seed 1.

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

SEED = 1

if __name__ == "__main__":
    sasrec_exp.main(
        max_users=300,
        max_movies=800,
        min_seq_len=5,
        top_k=10,
        gradisomap_epochs=300,
        cf_epochs=100,
        final_cf_epochs=50,
        lr_isomap=1e-3,
        n_run=f"sasrec_ml1m_latent32_300x800_seed{SEED}",
        seed=SEED,
        dataset_dir_name="ml-1m",
        dataset_type="movielens",
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
