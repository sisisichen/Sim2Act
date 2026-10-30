# Checkpoint Status

No trained Sim2Act, LocalMLP or LinearRanker checkpoints or fitted feature
scalers are distributed in this release. The frozen upstream OmniCrack30k
predictor is also not bundled. The source can support component-level research,
but randomly initialized models cannot reproduce the reported results.

The paper retained five model seeds, 2025–2029, for each learned method.
Sim2Act and LocalMLP selected epoch 1 for every retained seed. LinearRanker
selected epochs 11, 12, 3, 6 and 18, respectively. Models use 16 global-state
features and 24 candidate-local features with a scaler fitted only on the
100-scene development-training split.

Study-specific trained rankers, scalers and original split manifests may be
requested from the corresponding author, Yuyang Bao,
<yuyang.bao@outlook.com>, subject to applicable redistribution restrictions.
The manuscript's availability paragraph describes the status before this core
source release; it is not a checkpoint download link.

For any future checkpoint release, provide:

- The method, model seed, selected epoch, architecture and source commit.
- SHA-256 hashes for the checkpoint and its matching scaler.
- Exact development split identifiers and feature ordering/schema.
- Training command, loss settings, dependency versions and device details.
- Validation criterion and the checkpoint's provenance within the five-seed set.
- Redistribution terms for the weights and any dependent upstream artifacts.

Keep downloaded or newly trained artifacts outside version control unless they
are deliberately prepared for redistribution. A checkpoint must remain paired
with its training-fitted scaler; target-data scaler refitting changes the
evaluated protocol. See [Reproducibility](../docs/REPRODUCIBILITY.md).
