# Code map

Guide to the repository: modules, script families and arm groups. The top-level [README](../README.md) covers installation and reproduction; this page is the reference you open when you need to know what a particular script does.

## Repository structure

```
repo_paths.py                  every path in one place; importing it sets up sys.path
requirements.txt
docs/                          this guide and the documentation index
data/README.md                 the five datasets: links and expected layout
manul/                         the part of the MANUL library these experiments use
sasrec/, ncf/, shared/         architectures, trainers, pipelines, baselines, per-arm drivers
ncf_fast_and_nonparametric/    the second NCF pipeline: ML-1M GI-NCF and the non-parametric heads
analysis/                      metrics, geometry and topology diagnostics, tables
figures/                       scripts that produce every figure in the paper
results/                       computed artefacts (see results/README.md)
```

### `manul/` — the vendored part of the MANUL library

```
manul/Adam/Isomap.py           IsomapNN: differentiable Isomap. The learnable
                               parameter is the input distance matrix D_input;
                               the forward pass is Floyd-Warshall shortest
                               paths on a kNN graph, double centring, and an
                               eigendecomposition into latent coordinates Z.
manul/Adam/KernelPCA.py        the eigendecomposition IsomapNN builds on
manul/Adam/__init__.py
```

This is the only part of MANUL the recommender experiments touch
(`from Adam.Isomap import IsomapNN`). File and symbol names inside `manul/`
are left exactly as they are upstream; unused MANUL modules are not vendored.

### `shared/` — used by both model families

| file | role |
|---|---|
| `data_preparation_interactions.py` | per-user sequence construction, `max_users × max_items` subsampling, leave-one-out train/val/test split, item×user matrix construction |
| `idf_weighted_item_features.py` | IDF-weighted item features for the initial Isomap graph. Raw rating profiles make item-item distances nearly discrete, which creates geodesic twins, exact eigenvalue ties, and NaN gradients out of `torch.linalg.eigh`; IDF weighting removes the ties and lets the geometry train |

### `ncf/` — NeuMF / NCF family

**Architectures**

| file | role |
|---|---|
| `ncf_neumf_architecture.py` | the classic NeuMF (GMF + MLP branches) with real item embeddings — the Euclidean reference model |
| `ncf_neumf_on_manifold_architecture.py` | NeuMF whose item representations come from the Isomap coordinates `Z` through an item projection instead of an embedding table. With `freeze_item_projection=True` the projection is a fixed PCA map recomputed from the current `Z` at every outer step, so the model cannot launder a bad geometry through trainable projection weights |
| `ncf_negative_sampling_datasets.py` | future-blind training sampler and the 1-positive/99-negative evaluation sampler |
| `ranking_evaluation_ncf.py` | HR@K and NDCG@K for both the manifold and the plain model |

**Training**

| file | role |
|---|---|
| `gradient_isomap_ncf_bilevel_trainer.py` | the bilevel loop. Inner: train NeuMF-on-manifold on the current geometry. Outer: backpropagate the task loss into `D_input` through the Isomap layer and take one step. Saves `matrices_epoch*.npz` (`D_input`, `D_geodesic`, `knn_adj`, `Z`, `D_latent`) every outer epoch, a per-epoch `history.json`, and the restored-best checkpoint |
| `ncf_experiment_pipeline.py` | one complete run: load → subsample → split → optional pure-NCF baseline → bilevel training → final retrain → test evaluation → result JSON |
| `ncf_outer_loop_sweep_driver.py` | wraps the pipeline in a sweep over the outer step size `eta`, and runs the geometry diagnostics on each finished run |
| `queue_ncf_arts_all_scales_and_seeds.py` | a GPU-memory-gated queue that runs all scales × seeds of the Arts & Crafts dataset as separate subprocesses |

**Baselines and ablations**

| file | role |
|---|---|
| `baseline_ncf_euclidean_geometry_dump.py` | trains the Euclidean-geometry arm and dumps its item geometry under the same configuration as the results table |
| `baseline_ncf_poincare_fixed_geometry.py` | hyperbolic-MDS fit of the items into a Poincaré ball with a data-derived curvature, then NCF on that fixed geometry |
| `ablation_ncf_geometry_vs_optimization.py` | separates "the learned geometry is better" from "the bilevel loop trains the model better", by retraining from scratch on frozen geometry snapshots (initial / epoch 0 / epoch N) |

**`ncf/arms/` — one file per reported training arm**

Amazon Beauty in the paper configuration (`eta = 0.001`, frozen item
projection, dropout 0.2, selection by validation HR@10):
`arm_ncf_beauty_300x800_seed_arg.py` (seed as a CLI argument) plus
`arm_ncf_beauty_{1000x1200,3000x2500,6000x3000}_seed{0,1,2}.py`.
`arm_ncf_eta_sweep_{beauty,ml1m}_300x800.py` reproduce the outer-step-size
sweep. Each file is a thin wrapper that records the exact hyperparameters of
that arm — they are the configuration record, not just launchers.

### `sasrec/` — SASRec family

**Architectures**

| file | role |
|---|---|
| `sasrec_classic_architecture_and_data.py` | the classic SASRec stack (self-attention blocks, point-wise FFN, LayerNorm) with a real item embedding table, plus its train/eval datasets |
| `sasrec_on_manifold_architecture.py` | the same stack with the embedding table replaced by a projection of the Isomap coordinates `Z`, mirroring the NCF-on-manifold design including the frozen-PCA projection |
| `sasrec_manifold_samplers.py` | sequence sampler for training and the left-padded prefix dataset for evaluation |
| `ranking_evaluation_sasrec.py` | HR@K / NDCG@K for the sequential setting |

**Training**

| file | role |
|---|---|
| `gradient_isomap_sasrec_bilevel_trainer.py` | the SASRec bilevel loop. Same contract as the NCF trainer; because a full forward pass over all users does not fit in memory, the outer gradient `dL/dZ` is accumulated over one full batched pass |
| `sasrec_experiment_pipeline.py` | one complete SASRec run, end to end |

**Baselines and ablations**

| file | arm |
|---|---|
| `baseline_sasrec_classic_euclidean.py` | **Euclidean** — classic SASRec, configuration matched to the learned-geometry arm (maxlen 30, no dropout, no weight decay) |
| `baseline_sasrec_euclidean_geometry_dump.py` | dumps the classic arm's item geometry *from the same model the results table reports*, one tagged npz per dataset |
| `baseline_sasrec_poincare_mds_core.py` | the hyperbolic-MDS fit and the shared "train SASRec on a fixed `Z`" harness |
| `baseline_sasrec_poincare_data_curvature.py` | **Poincaré** — the fixed-geometry arm with a data-derived curvature and `latent_dim = 32` |
| `baseline_sasrec_hyperbolic_hypsasrec.py` | **Hyperbolic** — the HypSASRec-style variant |
| `ablation_sasrec_geometry_vs_optimization.py` | the geometry-versus-optimisation ablation, SASRec side |

**`sasrec/arms/` — one file per reported training arm**

* `arm_sasrec_latent32_*` — the adopted **"ours-32"** configuration
  (`latent_len = 32`, `hidden_units = 64`), MovieLens-1M and Amazon Beauty at
  three scales × three seeds.
* `arm_sasrec_latent32_any_dataset_scale_seed.py` — the same configuration
  parameterised by `--dataset / --scale / --seed`. The Arts, Grocery and
  Office cells of this arm were trained on a second machine; this script
  reproduces any cell of the full grid, including those.
* `arm_sasrec_latent64_*` — the earlier default `latent_len = 64`
  configuration, kept because the paper compares the two.
* `arm_sasrec_latent32_hidden32_beauty_300x800_seed{0,1,2}.py` — the
  `latent_len == hidden_units` ablation.

### `ncf_fast_and_nonparametric/` — the second NCF pipeline

Two NCF pipelines exist, and they are **not interchangeable**. The one in
`ncf/` produced the Amazon Beauty GI-NCF results and is what every SASRec
script shares its data preparation with. The one here is a later, vectorised
re-implementation that additionally supports non-parametric heads; it produced
the **MovieLens-1M GI-NCF grid** and the **non-parametric-head grids**
(ItemKNN, `bary_cos`, `bary_euclid`).

They are kept separate rather than merged because their shared modules
genuinely diverge — `data_preparation_interactions_fast.py` adds duplicate
removal and a different tie-break when selecting the top-k users and items, so
the two pipelines do not subsample a dataset to the same cells. Reproducing a
published number means using the pipeline that produced it.

| file | role |
|---|---|
| `ncf_and_nonparametric_experiment_pipeline.py` | the entry point, a full CLI: dataset, scale, trainer (`log` or `fast`), geometry options, and `--np-head` to swap NCF for a non-parametric head |
| `gradient_isomap_ncf_bilevel_trainer_fast.py` | the vectorised bilevel trainer |
| `ncf_neumf_on_manifold_architecture_fast.py` | its NeuMF-on-manifold model |
| `nonparametric_recommendation_heads.py` | **the non-parametric heads**: `item_knn` (cosine item-item kNN over `Z`), `bary_cos` (Rocchio-style barycentre of the user's positives, scored by cosine) and `bary_euclid`. No trainable parameters |
| `gradient_isomap_nonparametric_bilevel_trainer.py` | the bilevel loop for those heads: each epoch recomputes `Z` from `D`, scores, takes a BCE step on `D`; the best epoch is chosen by validation HR. Saves the same `matrices_epoch*.npz` snapshots as the other trainers |
| `vectorized_ranking_evaluation.py` | batched HR@K / NDCG@K / MRR |
| `data_preparation_interactions_fast.py`, `ncf_negative_sampling_datasets_fast.py`, `ranking_evaluation_ncf_fast.py` | this pipeline's own data and evaluation modules |
| `baseline_ncf_poincare_curvature_search_ml1m.py` | the Poincaré-curvature search on ML-1M |
| `structured_run_logger.py`, `determinism_seeding.py`, `check_training_determinism.py` | run logging, seeding, and a determinism check |
| `plot_ncf_training_curves.py` | loss / metric / geometry-displacement curves |

The `log` trainer option resolves to `ncf/gradient_isomap_ncf_bilevel_trainer.py`.
That file is a strict, backwards-compatible superset of the copy that shipped
on this pipeline's branch — the extra arguments (`grad_clip_norm`,
`matrices_save_every`, `warm_start_D_input`) all default to the previous
behaviour — so one copy serves both.

**`ncf_fast_and_nonparametric/arms/`**

| file | arm |
|---|---|
| `arm_gincf_ml1m_any_scale_seed.py` | **GI-NCF on MovieLens-1M**, `--scale` x `--seed`, in `paper` (the default: outer lr 0.001, frozen item projection, selection by validation HR@10, 300 outer epochs) or `paper-no-resample` (identical, negatives drawn once). The flag lists are transcribed from the finished runs' logs |
| `arm_nonparametric_head_grid.py` | replays any cell of the **non-parametric grid** from `results/tables/nonparametric_grid_configs.json` — 44 cells across 2 datasets, 2 heads and 3 scales, with per-cell hyperparameter combos; three seeds each. `--list` prints the grid, `--dry-run` prints commands |

### `analysis/` — metrics, geometry and topology

**The core measurement library**

`analysis/geometry_topology_diagnostics.py` computes, for one
`matrices_epoch*.npz` snapshot:

* **Ollivier-Ricci curvature** on the kNN graph (mean, median, negative
  fraction) — a local measure, via optimal transport between neighbourhoods;
* **graph Laplacian spectrum** — algebraic connectivity `lambda2`,
  `lambda_max`, spectral gap, number of connected components;
* **isometry of the embedding** — Kruskal stress, Spearman rho between
  geodesic and latent distances, residual variance;
* **persistent homology** (Vietoris-Rips, H0 and H1) on both the geodesic and
  the latent distance matrix.

`analyze_run()` walks a whole run, writes one CSV row per analysed epoch, and
checkpoints the CSV after every snapshot so a crash loses at most one epoch.
It uses worker processes: **any script calling it must keep its
`if __name__ == "__main__":` guard** — on Windows the children re-import the
module, and without the guard that is a fork bomb.

**Hyperbolicity and curvature**

| file | role |
|---|---|
| `gromov_delta_hyperbolicity.py` | exact Gromov four-point delta on a distance matrix |
| `dataset_delta_hyperbolicity_and_curvature.py` | **`delta_rel` and the curvature `c`** for MovieLens-1M and Amazon Beauty, three scales, following the HypSASRec procedure |
| `dataset_delta_hyperbolicity_extra_categories.py` | the same for Arts, Grocery, Office |
| `hyperbolicity_quality_table_core.py` | per-geometry `delta_rel` + diagnostics + quality, one row per geometry state |
| `hyperbolicity_quality_table_by_scale.py` | the same across all scales and all three geometries under equal conditions — writes the scaling table read by the hyperbolicity figure |
| `hyperbolicity_quality_table_generic.py` | the dataset-agnostic version of the above |

A note on where `delta` is defined: `D_input` is a free learnable parameter
with no triangle-inequality guarantee, so Gromov delta computed on it is
formally undefined. Report delta on `D_geodesic` (shortest paths, metric by
construction) and on `D_latent` (Euclidean).

**Diagnostics walkers** — `compute_diagnostics_*.py`, one per run family
(NCF paper configuration, ML-1M NCF and SASRec, SASRec latent-32 on the extra
categories, the IDF-initialised runs, the non-parametric heads, and a repair
script for runs whose CSV was written while training was still going).

**Tables and audits**

| file | role |
|---|---|
| `table_geometry_before_after_summary.py` | the before/after geometry table across every run family |
| `table_topology_before_after.py` | the topology counterpart |
| `table_sasrec_arm_metric_statistics.py` | **the statistics of the SASRec arms**: per-arm means, sign tests, per-cell winners and whether their margins exceed the seed spread, HR/NDCG agreement |
| `audit_topology_measure_reliability.py` | H1 coverage and which measures are actually populated |
| `compare_manifold_deformation_ncf_vs_sasrec.py` | how far each model moves the manifold, measure by measure |
| `compare_best_vs_last_checkpoint.py` | how much quality is lost between the best and the final epoch |
| `check_frozen_projection_cells.py` | detects cells where the geometry never moved |
| `persistent_h1_subsampled_large_scales.py` | subsampled H1 where the in-pipeline computation skips it |

**Evaluation-protocol checks** — `popularity_baseline_*.py` run a
most-popular, a random and an anti-popularity ranker under exactly the
project's evaluation protocol, and
`check_popularity_leak_in_eval_protocol.py` tests whether a scorer's advantage
comes from the leave-one-out split depressing the target item's training
count. These matter: under this protocol a trivial anti-popularity ranker
reaches far above chance at small scale, so small-scale cells cannot carry
claims on their own.

`probe_*.py` and `inventory_*.py` are inspection utilities that report what a
run tree actually contains (configurations, history lengths, snapshot counts)
before any aggregation is attempted.

### `figures/`

| file | figure |
|---|---|
| `plot_hyperbolicity_and_quality_by_scale.py` | **hyperbolicity/curvature versus quality across scales** — `delta_rel` against HR@10 for the learned, Poincaré and Euclidean geometries at every scale, plus quality versus scale per geometry |
| `curvature_rank_sensitivity_sweep.py` | how much the curvature `c` depends on the SVD rank it is estimated at |
| `plot_manifold_projection_panels.py` | **manifold visualisation from a distance matrix**: four item clouds side by side — data geometry → Euclidean → fixed Poincaré → learned — each projected from its own distance matrix, with the quality metric taken from the same artefact as the cloud above it |
| `plot_manifold_projection_panels_shaded.py` | the same with a KDE backdrop (the style used in the paper) |
| `plot_manifold_genre_territories_ml1m.py` | MovieLens-1M panels coloured by primary genre |
| `plot_manifold_clusters_in_original_space_ml1m.py` | the control that clusters in the original space rather than the projection |
| `assemble_sasrec_overview_table.py` | collects the four SASRec arms × 5 datasets × 3 scales into one table |
| `plot_sasrec_overview_figures.py`, `plot_sasrec_main_figure.py`, `plot_sasrec_results_table_figure.py`, `plot_sasrec_results_table_variants.py` | the SASRec results figures |
| `plot_training_history_curves.py`, `plot_nonparametric_history_curves.py` | training-history curves |
| `collect_ncf_and_nonparametric_metrics.py` | the NCF / non-parametric-head metric table |
| `export_appendix_metrics_tables_tex.py` | the appendix tables as LaTeX |

**On the manifold panels.** Two failure modes are worth knowing: a caption
metric must come from the same artefact as the cloud it labels (a fallback
filename pattern once captioned one dataset's panel with another dataset's
score), and the last snapshot in a run directory is not necessarily the final
geometry — check that a final model exists before treating a snapshot as
converged.
