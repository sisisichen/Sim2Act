# Sim2Act Model Card

Sim2Act learns candidate-view ordering for budget-constrained 2.5D crack mapping
on a registered surface with a fixed, known camera pool. This release accompanies
the 2026 manuscript **Simulation-supervised view ranking for budget-constrained
2.5D crack mapping** by Si Chen, Yuyang Bao, Longsheng Bao, Jiakang Zhao and
Ling Yu.

## Release Status

The repository provides core research source, manuscript figures, protocol
documentation and the [final Word manuscript](manuscript/Sim2Act_AIC_Manuscript1031.docx).
Trained rankers, fitted scalers, original split manifests, projected evidence
caches and benchmark datasets are not included. The released core is not a
complete executable reproduction of the historical benchmark pipeline. See
[Reproducibility](docs/REPRODUCIBILITY.md) and
[Checkpoints](checkpoints/README.md).

The manuscript's availability paragraph records the status before this source
release. The files actually present in this repository define the current
release scope; inclusion of source does not imply inclusion of the study's
trained artifacts or data.

## Task and Inputs

At each decision, the selector ranks the remaining candidate views using 16
global-state features and 24 candidate-local features. These summarize revealed
crack evidence, observation support, entropy, disagreement and known candidate
geometry. Candidate RGB, unrevealed predictions, reference labels, realized
candidate gains and dataset identity are excluded from predictive inputs.

After selecting a view, its cached evidence is revealed and fused by the frozen
two-class CS-Max backend. The public OmniCrack30k image predictor is held fixed.
Surface geometry and camera registration are supplied before selection; their
construction cost is outside the view budget.

## Architecture and Supervision

| Component | Manuscript configuration |
| --- | --- |
| Global encoder | `16 -> 64 -> 64` |
| Shared candidate encoder | `24 -> 96 -> 96` |
| Candidate-set context | Masked mean and maximum pooling, 192 dimensions |
| Shared scoring head | `352 -> 128 -> 64 -> 1` |
| Activations and regularization | GELU, layer normalization, dropout `0.1` |
| Trainable parameters | `70,785` |

Each remaining candidate is added independently to the same development state
to compute its numerical one-step surface-band IoU gain under CS-Max. Listwise,
pairwise and regression objectives supervise the ranking. The output is a
relative utility score, not a calibrated absolute gain or a stopping threshold.
The model and training-fitted scaler are frozen for evaluation.

## Data and Evaluation

The public CrackEnsembles validation release is partitioned into 100 development
training scenes and 28 development validation scenes. Final evaluation uses 208
held-out semi-synthetic scenes. Real evaluation uses 15 positive segments from
five CrackStructures structures; three reference-empty segments from one
additional structure form the negative controls. No real-data training,
fine-tuning, recalibration or checkpoint selection is performed.

The primary metric is normalized raw surface-band IoU AUC at a 10 mm reference
radius. Means average seeds within scenes; real results additionally average
three segments within each structure and weight the five structures equally.

| Method | Semi-synthetic raw AUC | Real raw AUC |
| --- | ---: | ---: |
| Coverage-NBV | 0.7592 | 0.8340 |
| LocalMLP | 0.8324 | 0.7172 |
| Sim2Act | 0.8305 | 0.6986 |

The paired semi-synthetic Sim2Act-minus-Coverage difference is `+0.071312`
(95% paired scene-bootstrap CI `[+0.051434, +0.090927]`). The corresponding
Sim2Act-minus-LocalMLP difference is `-0.001877` (CI `[-0.005432, +0.001354]`);
the evaluated configuration shows no resolved additional benefit over LocalMLP.
On the five real structures, the mean differences are `-0.135467` against
Coverage and `-0.018621` against LocalMLP. The reversal relative to Coverage
persists at reference radii of 5, 10 and 20 mm. These are manuscript results,
not measurements reproduced by installing this release.

## Intended Use and Limitations

The core supports research on sequential evidence prioritization for an already
registered asset surface. It has not demonstrated prospective field capture,
platform routing, structural diagnosis, physical crack-width measurement or an
operational digital twin.

Surface-band IoU measures agreement with a centerline neighborhood, not physical
crack area. It is distinct from tolerance-based centerline matching. CS-Max
may accumulate false positives, and raw map quality may decrease as views are
added. Best-so-far AUC and reference-target view counts are retrospective
diagnostics, not executable stopping rules or demonstrated field savings.

Transfer is limited by the observed real-domain reversal and the small number
of independent real structures. Every deployable selector produces false
positives on the single negative structure. The causes of the transfer reversal
remain unresolved. Selector computation also costs more than Coverage-NBV in
the recorded experiments; capture, travel and communication costs were not
measured.

## Contact and Citation

Corresponding author: Yuyang Bao, <yuyang.bao@outlook.com>.
Use [CITATION.cff](CITATION.cff) for manuscript citation metadata.
