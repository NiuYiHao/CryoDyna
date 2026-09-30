# 80S pose-search control and the 1AKE failure case

(a,b) Fixed external 80S density, first 64 prepared synthetic particles, seed 0.
All particles are retained through coarse Hopf search, four refinements and 30
Adam passes. Adam pass 0 repeats the last search estimate. Axes have different
ranges. The evaluator chooses one global frame correction and whole-dataset
hand per stage using min(100,N) candidate corrections and mean Frobenius error;
fit and evaluation use the same 64 particles. Viewing-direction and complete
SO(3) angles are separate metrics. The reference volume is an input, so this
control measures pose recovery conditional on a known density.

(c,d) 1AKE reference-structure Optpose (128 training particles, 5 passes), learned
volume DRGN-AI (50,000 training particles, 31 passes) and Haar random SO(3).
The same 25 calibration particles fit each global frame/hand; the other 103
particles supply all plotted means. Actual training durations and priors differ.
All three endpoints remain far above the mean ≤5° accuracy target. Each method
has one measured trajectory; uncertainty over training repetitions remains pending.

The 80S DRGN-AI replication was in its first HPS epoch at this release snapshot.
Its measured curve/endpoints are pending. The published Fig. 1c uses viewing-direction
error (six runs), DOI: https://doi.org/10.1038/s41592-025-02720-4.
Sub-degree accuracy of the fixed-density control establishes a conditional
pose-fitting result; strict neural ab-initio replication remains an open gate.

Rebuild: `python scripts/figure_optpose_release.py` from the repository root.
The source tables and SHA256 records accompany this figure.
