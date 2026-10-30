# Primary raw AUC results

`primary_raw_auc.csv` contains the raw surface-band IoU AUC columns matching
Tables 3 and 4 of the October 31 manuscript. Semi-synthetic evaluation uses
208 held-out scenes. Real evaluation covers 15 positive segments within five
structures. The default reference-band radius is 10 mm.

Values are copied from the study result tables at full stored precision;
the repository README displays the manuscript's four-decimal values.
Oracle-NBV is a privileged greedy reference. Normalization by the all-view
endpoint permits AUC values above one.

The final manuscript gives the paired semi-synthetic Sim2Act-minus-Coverage
raw-AUC difference as +0.071312, with 95% scene-bootstrap interval
[+0.051434, +0.090927]. Its Sim2Act-minus-LocalMLP difference is -0.001877,
with interval [-0.005432, +0.001354]. Older audit tables containing envelope
statistics are not included in this release.

These aggregate values do not replace per-initialization raw trajectories,
structure grouping, or the complete evaluation artifacts needed for reruns.
