"""GI-NCF on Amazon Beauty, 1000 users x 1200 items, seed 2.

Paper configuration: outer step size eta=0.001, frozen item projection,
dropout 0.2, checkpoint selection by validation HR@10, inner and final CF
stages capped at 200 epochs with patience 30.

This arm: 300 outer epochs.

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
import ncf_outer_loop_sweep_driver as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="amazon_beauty_frozenproj_scaletest_1000x1200_seed2",
        max_users=1000,
        max_movies=1200,
        dataset_dir_name="amazon_beauty",
        dataset_type="amazon",
        amazon_category="Beauty_and_Personal_Care",
        select_by="hr",
        cf_epochs=200,
        final_cf_epochs=200,
        inner_patience=30,
        final_patience=30,
        outer_epochs=300,
        eta_values=[0.001],
        dropout=0.2,
        freeze_item_projection=True,
        seed=2,
        diagnostics_workers=4,
        summary_filename="amazon_beauty_frozenproj_scaletest_1000x1200_seed2_summary.json",
    )
