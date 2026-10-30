# Reproducibility

This release exposes core Sim2Act research source, baseline components,
manuscript figures and protocol records. It does not include the full historical
data preparation and experiment orchestration environment. Installing the core
package or passing its focused tests does not reproduce the paper's metrics.
See the root [README](../README.md) for the commands and dependencies supplied
with this release. [Source provenance](SOURCE_PROVENANCE.md) identifies the
included components and their verification. The core `trainer.train_smoke`
helper is a bootstrap check, not the final paper's complete training
orchestration.

## Available and Required Materials

| Available in this repository | Required separately for full paper reproduction |
| --- | --- |
| Core source and focused tests | Original benchmark scenes and annotations |
| Manuscript and extracted figures | Pinned upstream predictor and mapping/extraction pipeline |
| Paper configuration record | Exact development/test split manifests |
| Model and dataset protocol documentation | Projected sparse evidence caches and numerical metadata |
| Curated manuscript result summaries | Historical ranker checkpoints, fitted scalers and locked trajectories |

The [paper configuration](../configs/paper_sim2act.json) records confirmed
manuscript settings. It is protocol metadata, not an executable configuration
for a complete benchmark workflow. Study-specific trained artifacts and
original manifests may be requested from the corresponding author, subject to
third-party redistribution restrictions. See [checkpoint status](../checkpoints/README.md).

The Word manuscript's availability paragraph predates this source release.
The release adds the files checked into this repository; it does not retroactively
change the manuscript or imply that all experimental artifacts are public.

## Historical Experiment Environment

For the released core modules, install the package from the repository root:

```bash
python -m pip install -e .
```

Install PyTorch for the intended CPU/CUDA platform before adding the neural
and test dependencies:

```bash
python -m pip install -e ".[learning,test]"
python -m pytest
```

The `science` extra declares the additional cache and geometry dependencies:

```bash
python -m pip install -e ".[science]"
```

PyTorch3D and the upstream CrackStructures, ENSTRECT and OmniCrack30k projects
must be installed separately using their compatible upstream environments.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the project links.
Optional dependency bounds are not a complete historical environment lock.

The released neural factory is
`sim2act_baselines.sim2act.step5b_models.make_model`. Feature extraction,
one-step supervision, training helpers and ranking evaluation reside in
`feature_extractor.py`, `exact_candidate_gain.py`, `trainer.py` and
`evaluator.py` in that same package. See the
[source provenance record](SOURCE_PROVENANCE.md) for exact scope and the
verification actually performed on this release.

The main experiments used Python 3.10, PyTorch 2.3.1 with CUDA 12.1, PyTorch3D
0.7.7, NumPy 1.26.4 and Open3D 0.19.0 on one NVIDIA RTX 4090 D. SciPy and
scikit-learn were also used; the manuscript does not pin their versions.
Post-lock CPU training diagnostics and reconstructed untrained controls were
separate from the historical GPU evaluation artifacts and did not replace them.

Record the release commit, exact dependency versions, operating system, device,
command and random seeds for any new run. New environments and synthetic test
inputs should be identified explicitly rather than presented as historical
paper artifacts.

## Numerical Contract

Full reproduction requires more than matching the ranker dimensions:

- Use the same sampled surface, registered cameras, sparse evidence and candidate
  ordering. Each scene begins with three shared random views.
- Preserve CS-Max's two-class weights `[1, 10]`, float16 weighted fusion and
  background assignment on exact score ties. Stored fused scores are float32.
- Preserve the inclusive `[90, 270]` degree eligibility rule: fusion uses angles
  cast to float16, while observation counts and probability moments use original
  float32 angles.
- Build the 16 global and 24 candidate features exclusively from revealed
  evidence and known geometry. Use the training-fitted scaler with standard
  deviation floor `1e-6`, clip standardized features to `[-8, 8]`, and mask
  candidate padding.
- Compute supervision by adding each remaining candidate independently to the
  same state. Retain negative gains and preserve all deterministic tie rules.

All five declared nonlinear model seeds selected epoch 1 in the historical
runs, followed by 12 epochs without improvement. Checkpoint selection minimized
unrounded scene-balanced one-step validation regret; ties favored NDCG@5,
Oracle top-3 hit rate and then the earlier epoch. This validation rule is
distinct from closed-loop raw-AUC evaluation.

## Primary Metric and Statistical Unit

Raw surface-band IoU AUC is primary. For each trajectory, trapezoidal integration
uses every integer budget from three to all `N` views, with normalized budget
`b = k/N`. Divide the integral by that scene's all-view IoU and by the integration
span `1 - 3/N`. AUC may exceed one because a subset can outperform the all-view
CS-Max endpoint. Zero-reference negative controls are analyzed separately.

Average learned-model seeds and initializations within each scene before
population aggregation. Weight semi-synthetic scenes equally; for real data,
average segments within structure and weight the five structures equally.
The primary semi-synthetic paired intervals use 5,000 bootstrap resamples of
208 scene differences. They condition on retained checkpoints and recorded
initializations and do not quantify uncertainty from new training runs.

Best-so-far AUC, views to 90% of all-view reference and same-state Oracle regret
are secondary offline diagnostics. Their confidence intervals must not be
relabelled as raw-AUC intervals. For real data, leave-one-structure-out ranges
are descriptive sensitivity summaries, not confidence intervals; the paper
does not report raw-AUC p values for the five structures.

## Geometry and Runtime Interpretation

Keep `legacy` and `geometry_small_cluster_v1` extraction artifacts separate.
The Delaunay-MST replacement was audited against the complete-graph MST, while
the small-cluster amendment changes downstream processing. Identical semantic
labels can therefore yield different geometric scores across extraction
versions. Do not pool those versions into a policy comparison.

Recorded selector time is cumulative over a complete acquisition trajectory.
Prediction-trajectory timings use a different population and are not component
timings that can be subtracted. Capture, travel, communication, initial
reconstruction and camera registration costs were outside the experiment.
