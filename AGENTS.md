# AGENTS.md

Operating guide for coding agents working in this repository. Humans should
read `README.md` first — this file only adds what an agent needs in order not
to break things. Everything here is a rule about *this* codebase, not generic
advice.

## What this repository is

Reproduction code for a paper on **learned item geometry in recommendation**.
Two recommender architectures (NeuMF/NCF and SASRec) plus non-parametric heads
are trained on top of an item geometry that is itself optimised, by a bilevel
loop whose outer variable is the input distance matrix of a differentiable
Isomap layer. Each is compared against fixed Euclidean, fixed Poincaré and
hyperbolic geometries, across five datasets and four sample scales, and the
resulting manifolds are measured with curvature, spectral, isometry and
persistent-homology diagnostics.

It is a **research reproduction package, not a product**. Numerical fidelity
to the published runs outranks elegance every time.

## Setup

```bash
pip install -r requirements.txt
```

Datasets are not in the repo; see `data/README.md`. Nothing but the figure and
table scripts can run without them, and training needs a CUDA GPU.

## Rules that are easy to violate

**Never hardcode an absolute path.** Every location comes from `repo_paths`,
which also puts all source directories on `sys.path`. New scripts start with:

```python
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))   # ".." per nesting level
import repo_paths as _paths  # noqa: E402,F401
```

Then use `_paths.DATA_DIR`, `_paths.RUNS_DIR`, `_paths.RESULTS_DIR`,
`_paths.TABLES_DIR`, `_paths.DIAGNOSTICS_DIR`, `_paths.FIGURES_OUT_DIR`,
`_paths.EXTERNAL_RUNS_DIR`. Add a new location to `repo_paths.py` rather than
computing it inline.

**`if __name__ == "__main__":` is mandatory in anything that reaches
`analyze_run`.** It forks workers; on Windows the children re-import the
module, and an unguarded top-level call is a fork bomb that silently re-runs
whole training pipelines. This has happened and cost about twenty GPU-hours.
Every arm script and every diagnostics walker must keep its guard.

**Two NCF pipelines exist and are not interchangeable.** `ncf/` is the
reference pipeline (Amazon Beauty GI-NCF; shares data preparation with every
SASRec script). `ncf_fast_and_nonparametric/` is a vectorised
re-implementation that also carries the non-parametric heads (ML-1M GI-NCF,
ItemKNN, bary_cos). Their `data_preparation_*` modules genuinely diverge —
duplicate removal, and a different tie-break when picking the top-k users and
items — so the two do **not** subsample a dataset to the same cells. Do not
"deduplicate" them into one module. Reproducing a number means using the
pipeline that produced it.

**Flat imports, deliberately.** Modules import each other by bare name
(`from data_preparation_interactions import ...`) because `repo_paths` puts
every source directory on `sys.path`. Renaming a module means updating its
import sites; there is no package namespace to hide behind.

**`manul/` is vendored.** It is the subset of an external library these
experiments use (the differentiable Isomap layer and its kernel-PCA
dependency). Do not rename files or symbols inside it, and do not add to it.

**Arm scripts are configuration records.** Each file in `*/arms/` pins the
exact hyperparameters of one reported run. Their flag lists were transcribed
from the finished runs' logs. Do not "tidy" them, merge them, or change a
value to something that looks more sensible — if a config looks wrong, say so,
do not silently fix it.

## Output goes to `runs/`, never next to the code

Training writes per-epoch logs, `matrices_epoch*.npz` geometry snapshots,
checkpoints and result JSONs under `repo_paths.RUNS_DIR`. Analysis reads
snapshots and writes CSVs under `repo_paths.RESULTS_DIR`. Figures land in
`repo_paths.FIGURES_OUT_DIR`. All four are gitignored. `EXTERNAL_RUNS_DIR`
holds run trees produced elsewhere and is **read-only** — never write into it.

## Data traps that have produced wrong conclusions

These are not hypothetical; each one has already caused a wrong number.

- **`h1_count = -1` is a sentinel meaning "skipped for safety", not "zero
  cycles".** Persistent homology bails out above `max_simplices = 300_000`,
  which happens from roughly 2500 items up. Filter `>= 0` before aggregating,
  or epoch-0 sentinels poison the means.
- **The outer step size changes the answer qualitatively.** Runs at a large
  `eta` are not "the same search, faster" — several measures move in the
  opposite direction. Never pool runs with different `lr_isomap`. Filter on
  the run name first.
- **A truncated diagnostics CSV is never repaired automatically.** The walkers
  skip any run that already has a CSV, so a CSV written while training was
  still going stays truncated forever and understates how far the manifold
  moved. Move it aside by hand.
- **The last snapshot in a run directory is not necessarily the final
  geometry.** Check that a final model exists before treating a snapshot as
  converged.
- **A caption metric must come from the same artefact as the thing it
  labels.** A fallback filename pattern once captioned one dataset's manifold
  panel with another dataset's score.
- **Gromov delta is defined on a metric space.** `D_input` is a free learnable
  parameter with no triangle-inequality guarantee, so delta on it is formally
  undefined. Report delta on `D_geodesic` (shortest paths, metric by
  construction) and `D_latent` (Euclidean).
- **Small scales cannot carry claims on their own.** Under this evaluation
  protocol a trivial anti-popularity ranker scores far above chance at the
  smallest scale. `analysis/popularity_baseline_*.py` and
  `analysis/check_popularity_leak_in_eval_protocol.py` exist to check this.

## Before claiming a result

Compute the statistic. Do not read a trend off a plot and describe it as
"improved", "fixed" or "overfitting" — compute the correlation, the
difference, the p-value, and report n. Single-seed before/after deltas on this
pipeline are not reliable enough to call a verdict; a single-seed pilot has
already inverted a published conclusion that three seeds then reversed back.

Compare invariant properties of the manifolds rather than raw value
correlations, and frame findings as characteristics of a configuration, not as
failures.

## Verifying a change

There is no test suite. The standing check is that everything parses, imports,
and that nothing identifying crept in:

```bash
# every file compiles and every module imports
python - <<'EOF'
import importlib, importlib.util, io, os, glob, sys
sys.path.insert(0, "."); import repo_paths
bad = []
for dp, dn, fn in os.walk("."):
    if ".git" in dp: continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            try: compile(io.open(p, encoding="utf-8").read(), p, "exec")
            except Exception as e: bad.append((p, e))
for d in ("analysis", "figures", "ncf", "sasrec", "shared",
          "ncf_fast_and_nonparametric"):
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            try: importlib.import_module(f[:-3])
            except Exception as e: bad.append((d + "/" + f, e))
for p in sorted(glob.glob("*/arms/*.py")):
    spec = importlib.util.spec_from_file_location("t_" + os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(m)       # __main__ guard keeps this cheap
    except Exception as e: bad.append((p, e))
print("problems:", len(bad)); [print(" ", p, repr(e)[:120]) for p, e in bad]
EOF

# a real end-to-end check that needs no data and no GPU
python figures/plot_hyperbolicity_and_quality_by_scale.py
```

Loading an arm script must be free of side effects — if executing one starts
training, its `__main__` guard is missing.

## Style and content constraints

- **English only**, everywhere: code, comments, docstrings, printed output,
  committed data files.
- **No personal or machine-specific content.** No user names, home
  directories, drive letters, UNC paths, host names, internal URLs, references
  to colleagues or to who ran what, and no pointers to private notes,
  scratchpads or chat sessions. Anything a reader outside the project cannot
  resolve does not belong in the repository.
- File names are self-describing: `plot_manifold_projection_panels.py`, not
  `panels4.py`. Follow the existing prefixes — `arm_`, `baseline_`,
  `ablation_`, `compute_diagnostics_`, `table_`, `plot_`, `probe_`, `check_`.
- Match the surrounding comment density. These files explain *why* a choice
  was made, often with the measurement that forced it. Keep that when editing
  near such a comment; do not delete the reasoning to shorten a diff.

## Scope

Do not add datasets, models or arms that the paper does not report. Do not
commit training artefacts or data. If something looks like a bug in a pinned
configuration, report it rather than changing it — the configurations are
evidence.
