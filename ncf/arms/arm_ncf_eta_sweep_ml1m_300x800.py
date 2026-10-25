"""GI-NCF on MovieLens-1M, 300 users x 800 items.

Sweep over the outer step size eta at a fixed scale. The geometry the outer
loop converges to depends qualitatively on eta, so these runs are reported
separately and must never be pooled with the eta=0.001 arms.

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
        n_run_prefix="eta_sweep_p30",
        max_users=300,
        max_movies=800,
        dataset_dir_name="ml-1m",
        dataset_type="movielens",
        select_by="hr",
        cf_epochs=200,
        final_cf_epochs=200,
        inner_patience=30,
        final_patience=30,
        summary_filename="eta_outer_sweep_p30_summary.json",
    )
