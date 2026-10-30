# Dataset Protocol

This document records the protocol in Sections 3, 4 and 6 of the
[final manuscript](../manuscript/Sim2Act_AIC_Manuscript1031.docx). Datasets,
annotations, registered evidence caches and the original split manifests are
not distributed in this core source release.

## Public Sources and Splits

| Source / partition | Scenes | Role |
| --- | ---: | --- |
| CrackEnsembles development training | 100 | Oracle-label generation, scaler fitting and model training |
| CrackEnsembles development validation | 28 | Checkpoint selection |
| CrackEnsembles held-out test | 208 | Final semi-synthetic evaluation |
| CrackStructures positive | 15 | Real evaluation, five structures with three segments each |
| CrackStructures negative controls | 3 | Reference-empty segments from one additional structure |

The upstream CrackEnsembles publication reports 1,090 training, 128 validation
and 211 test scenes. The reported training scenes were unavailable in the public
release when this study's protocol was frozen. The 128 available validation
scenes were therefore partitioned deterministically by SHA-256 into 100/28
development scenes. Three test scenes previously used for pipeline diagnostics
were excluded, leaving 208 final-test scenes. Counts alone cannot reconstruct
the exact historical memberships; the original manifests are required.

The positive CrackStructures structures are `beam`, `bridge_b`, `bridge_s`,
`indoors` and `wall`. Positive candidate pools contain 9–56 views. The negative
controls are three `bridge_g` segments with 31, 28 and 16 views. Segments and
repeated seeds are not independent assets.

## Evidence and Reference Representation

Each supplied mesh is sampled at 1,000,000 surface points using the pinned
ENSTRECT sampler and its internal seed 42. A frozen public OmniCrack30k nnU-Net
checkpoint supplies per-view probabilities, projected using the supplied
cameras and surface geometry. Methods share the evidence, class weights and
semantic hardening; no method-specific image threshold or recalibration is used.

The downloaded archives supply `annotations/crack.obj` centerlines rather than
native point-label PLY files. The reference centerline is interpolated at
spacing `0.001` and converted to a surface mask using a Euclidean neighborhood
radius of `0.010` under the pipeline's metre-coordinate convention. These
correspond to 1 mm sampling and a 10 mm radius. This is a positional reference
band, not measured physical crack area or width. Sensitivity evaluation uses
5, 10 and 20 mm radii on the same locked sequences.

Surface-band IoU counts TP, FP and FN over all fixed sampled surface points,
including unobserved reference-positive points. Official centerline
`clCloudIoU` uses a different counting domain and must be reported separately.

## Pre-reveal Information Boundary

Before a decision, selectors may use the revealed state and known candidate
geometry, including pose, visibility and redundancy. Candidate RGB, unrevealed
probabilities, reference labels and realized candidate gains are excluded from
predictive inputs. The selected candidate's cached evidence is revealed only
after selection, then fused by CS-Max.

Development reference labels generate candidate-level one-step gains. The
training corpus contains 11,229 states and 210,688 candidate labels; validation
contains 1,544 states and 28,409 labels. The scaler is fitted only on development
training states. Model development excludes final-test images, evidence and
annotations, including CrackStructures caches. Deployable evaluation
trajectories are locked before reference-based scoring and privileged Oracle
replay. Real data receive no model fitting or checkpoint selection.

## Aggregation and Redistribution

Evaluation uses ten common three-view initialization seeds, 2025–2034, and five
retained model seeds, 2025–2029. Semi-synthetic summaries weight 208 scenes
equally. Real summaries average the three segments within each structure, then
weight five structures equally. Negative controls are reported separately from
positive-scene normalized AUC.

Obtain CrackStructures, CrackEnsembles, OmniCrack30k and ENSTRECT from their
original projects and follow their source licenses. Third-party images and
annotations retain their original rights. This repository does not grant
additional rights to upstream datasets, pretrained predictors or annotations.
