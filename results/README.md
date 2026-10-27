# Computed results shipped with the repository

Everything here was produced offline from saved training artefacts — no GPU
and no retraining. It is included so that the table and figure scripts can be
run without first reproducing several hundred GPU-hours of training.

```
measures/   dataset-level quantities: delta-hyperbolicity, curvature, H1
tables/     aggregated metric / geometry / topology tables
diagnostics/ (not shipped) per-run, per-epoch geometry CSVs
```

## measures/

| file | contents |
|---|---|
| `hypsasrec_curvature.json` | `delta_rel` and the derived curvature `c` for MovieLens-1M and Amazon Beauty, three scales, binary and rating-weighted variants |
| `delta_new_datasets.json` | the same for Arts & Crafts, Grocery & Gourmet Food, Office Products |
| `h1_large_scales.json` | H1 cycle counts and maximum persistence at 3000×2500 and 6000×3000, where the in-pipeline computation skips H1 |

Curvature follows the HypSASRec procedure: item representations are the
rank-`d` truncated SVD of the binary training interaction matrix, `delta_rel =
2·delta/diam` is estimated over random item subsets (10 draws of 1500, or the
full set when smaller), and

    c = (delta_rel(Poincare disk) / mean delta_rel(X))^2

with the Poincare anchor at tolerance 1e-12. **These values are computed at
rank 32 and are valid only for a 32-dimensional geometry**: `delta_rel`
depends on the rank and so does `c`. A rank sweep on Arts 300×800 gives
`delta_rel` 0.400 at rank 8, 0.300 at 32 and 0.235 at full rank — `c` varies
threefold across that range. Reproduce the sweep with
`figures/curvature_rank_sensitivity_sweep.py`.

## tables/

| file | provenance / produced by |
|---|---|
| `full_hyperbolicity_table_frozenproj.csv` | the paper-config (Beauty 300×800) hyperbolicity/quality comparison table; read by `figures/plot_hyperbolicity_and_quality_by_scale.py` |
| `full_hyperbolicity_table_scaling.csv` | `analysis/hyperbolicity_quality_table_by_scale.py`; read by `figures/plot_hyperbolicity_and_quality_by_scale.py` |
| `overview_data.json` | `figures/assemble_sasrec_overview_table.py` (collector; writes next to the script) — four SASRec arms × 5 datasets × 3 scales, seeds 0–2; the shipped copy here is what the overview figure reads |
| `sasrec_metrics_table.csv` | shipped SASRec arm metrics table; read by the main results figure, appendix export and arm-statistics scripts |
| `summary_before_after.csv`, `summary_before_after.md` | `analysis/table_geometry_before_after_summary.py` |
| `topology_before_after.csv`, `topology_tables.md` | `analysis/table_topology_before_after.py` |
| `appendix_metrics_tables.tex` | `figures/export_appendix_metrics_tables_tex.py` |
| `diag_np_*.csv` | the non-parametric-head diagnostics scripts |
| `np_amazon_inventory.csv` | `analysis/inventory_nonparametric_amazon_runs.py` |
| `ncf_np_metrics_raw.csv` | shipped raw metrics sheet used as input by the collector |
| `ncf_np_metrics_long.csv` | `figures/collect_ncf_and_nonparametric_metrics.py` |
| `delta_rank_sweep.json` | `figures/curvature_rank_sensitivity_sweep.py` (cache) |
| `metrics_audit.txt` | a free-text inventory of which HR/NDCG cells exist in which run tree |

Rows whose provenance is a run tree from another machine carry the
placeholder prefix `EXTERNAL_RUNS/`, which corresponds to
`repo_paths.EXTERNAL_RUNS_DIR`.

## diagnostics/ — deliberately not shipped

The per-run, per-epoch geometry CSVs (one row per analysed epoch, 21 measures)
total about 1.3 GB and are therefore excluded. Regenerate them with the
`analysis/compute_diagnostics_*.py` walkers once the corresponding training
runs exist under `runs/` (or `external_runs/`); every downstream table script
reads them from `repo_paths.DIAGNOSTICS_DIR`.

## Two traps worth knowing before analysing any of this

**The eta-sweep trap.** A sweep over the outer step size `eta` produces runs
that look like the paper configuration but are not: at `eta` in 0.01–0.1, ten
to a hundred times the paper's `eta = 0.001`, several measures move in the
*opposite* direction (algebraic connectivity, Kruskal stress, latent H0
persistence, geodesic H1 persistence). Pooling the two sets yields convincing
p-values for conclusions that are wrong for the paper's configuration. Filter
on the run name before aggregating. That the converged geometry depends
qualitatively on the outer step size is itself a reported finding — large
`eta` is not "the same search, faster".

**The H1 sentinel.** `h1_count = -1` means "skipped for safety", not "zero
cycles" — `persistent_homology()` bails out when the estimated triangle count
exceeds `max_simplices = 300_000`, which happens from roughly 2500 items up.
Filter on `>= 0` before aggregating, or epoch-0 sentinels silently poison the
means. The subsampled estimates in `measures/h1_large_scales.json` cover that
gap; they are comparable across epochs within a scale, not across scales.

**A third, smaller one:** a run analysed while its training job was still
writing produces a truncated CSV, and the diagnostics walkers skip any run
that already has a CSV — so a truncated file is never replaced automatically
and has to be moved aside by hand. Truncated runs understate how far the
manifold moved.
