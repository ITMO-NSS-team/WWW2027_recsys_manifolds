"""Single place where every path used by this repository is defined.

Importing this module has two effects:

1. all source directories of the repository are put on ``sys.path``, so that
   the flat ``from <module> import ...`` style used throughout the code works
   from any working directory;
2. the directory constants below become available as
   ``repo_paths.<NAME>``.

Every location can be redirected with an environment variable, which is the
intended way to point the code at a large scratch disk or at a shared
network volume instead of the checkout itself:

===========================  ==========================  ====================
constant                     environment variable        default
===========================  ==========================  ====================
``DATA_DIR``                 ``RM_DATA_DIR``             ``<repo>/data``
``RUNS_DIR``                 ``RM_RUNS_DIR``             ``<repo>/runs``
``EXTERNAL_RUNS_DIR``        ``RM_EXTERNAL_RUNS_DIR``    ``<repo>/external_runs``
``RESULTS_DIR``              ``RM_RESULTS_DIR``          ``<repo>/results``
``FIGURES_OUT_DIR``          ``RM_FIGURES_DIR``          ``<repo>/figures_out``
===========================  ==========================  ====================

``RUNS_DIR`` holds everything a training job writes (per-epoch logs, the
``matrices_epoch*.npz`` geometry snapshots, checkpoints, result JSONs).
``EXTERNAL_RUNS_DIR`` holds run trees produced on another machine - the
analysis scripts read from it and never write to it.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env(var, default):
    return os.path.abspath(os.environ.get(var) or default)


# ---------------------------------------------------------------- source
MANUL_DIR = os.path.join(REPO_ROOT, "manul")        # vendored MANUL subset
SHARED_DIR = os.path.join(REPO_ROOT, "shared")
NCF_DIR = os.path.join(REPO_ROOT, "ncf")
NCF_FAST_DIR = os.path.join(REPO_ROOT, "ncf_fast_and_nonparametric")
SASREC_DIR = os.path.join(REPO_ROOT, "sasrec")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "analysis")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

# ------------------------------------------------------------- input data
DATA_DIR = _env("RM_DATA_DIR", os.path.join(REPO_ROOT, "data"))

# ------------------------------------------------- training run artefacts
RUNS_DIR = _env("RM_RUNS_DIR", os.path.join(REPO_ROOT, "runs"))
NCF_RUNS_DIR = os.path.join(RUNS_DIR, "logs_movielens_isomap_cf")
SASREC_RUNS_ROOT = os.path.join(RUNS_DIR, "sasrec")
SASREC_RUNS_DIR = os.path.join(SASREC_RUNS_ROOT, "logs_sasrec_isomap")
HYP_SASREC_RUNS_DIR = os.path.join(RUNS_DIR, "hyp_sasrec")

# Run trees produced elsewhere, e.g. on another machine. Read-only
# for every script in this repository.
EXTERNAL_RUNS_DIR = _env("RM_EXTERNAL_RUNS_DIR",
                         os.path.join(REPO_ROOT, "external_runs"))

# -------------------------------------------------------- computed output
RESULTS_DIR = _env("RM_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
DIAGNOSTICS_DIR = os.path.join(RESULTS_DIR, "diagnostics")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MEASURES_DIR = os.path.join(RESULTS_DIR, "measures")
FIGURES_OUT_DIR = _env("RM_FIGURES_DIR", os.path.join(REPO_ROOT, "figures_out"))

_SOURCE_DIRS = (MANUL_DIR, SHARED_DIR, NCF_DIR, NCF_FAST_DIR, SASREC_DIR,
                ANALYSIS_DIR, FIGURES_DIR, REPO_ROOT)

for _d in _SOURCE_DIRS:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def ensure_output_dirs():
    """Creates the output directories. Input directories are never created."""
    for d in (RUNS_DIR, RESULTS_DIR, DIAGNOSTICS_DIR, TABLES_DIR,
              MEASURES_DIR, FIGURES_OUT_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:          # read-only target: the caller will notice
            pass


ensure_output_dirs()
