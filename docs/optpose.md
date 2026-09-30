# CryoDyna-optpose: reproducible 1AKE pose benchmark

Measured results and remaining release gates: [acceptance record](OPTPOSE_ACCEPTANCE.md).

The CLI prepares controlled pose perturbations, runs the original and optimized
models, records per-particle predictions, and generates learning curves and
endpoint comparisons. Scientific acceptance is reported separately from software
tests. The initial acceptance targets are a mean full SO(3) error of at most 5° and
a 90th percentile of at most 10° for recovery without supplied poses.

## Environment and inputs

Use the repository installation instructions, then install
`requirements-optpose.txt` in that environment. DSSP (`mkdssp` or `dssp`) must be
available in its `bin` directory. Run the commands below from the repository root.
The implementation uses the local CryoDyna SO(3) grid and has no runtime dependency
on an ignored `develop/` checkout. The optional HPS parity tests use a local
DRGN-AI checkout when available.

The supplied `projects/atom_configs/1ake.py` identifies the 1AKE particle STAR,
image stacks, pixel size and reference structure. The evaluation PDB directory
contains `1akeA_1.pdb` through `1akeA_50.pdb`. The STAR's `rlnClassNumber` maps each
particle to its simulated structure. Atom identities and order are validated.
All 50,000 particles are used by default. A fixed 20% calibration subset fits the
global ab-initio coordinate gauge; the remaining 80% supplies reported metrics.
This is transductive pose recovery on the supplied images, not a claim about
generalization to unseen images. Ground-truth poses and structures are used only
to create controlled perturbations and calculate metrics, never as training loss
targets. Ab-initio training receives a STAR containing image and CTF fields only.

## Run the perturbation experiment

```bash
python scripts/benchmark_optpose.py run --outdir results/optpose/full \
  --methods original opt --angles 0 5 10 15 20 --seed 1
```

`--angles` accepts any distinct array in [0,180] degrees, for example
`--angles 0 2.5 7.5 30` or `--angles '[0,2.5,7.5,30]'`. Each particle receives an independent uniformly random
axis and the exact requested geodesic rotation magnitude. Both methods use the
same axes, data order, reference warmup and epoch budget. Initial structural
tensors (including legacy attention layers) are compared at atol=1e-6, rtol=1e-6
to accommodate float32 rounding; their exact hashes are retained for provenance.
`--seed`, `--sample-seed`, `--epochs`, `--batch-size`, `--eval-batch-size`,
`--init-steps`, `--pose-lr`, `--pose-model table|head|hybrid` and
`--checkpoint-every` are explicit. `--rotation-only` fixes input translations
for a rotation-only ablation. The default permits translation refinement.

Pilot example (a pilot is labelled by its actual particle count):

```bash
python scripts/benchmark_optpose.py run --outdir results/optpose/pilot \
  --particles 1024 --epochs 20 --angles 5 20 --checkpoint-every 20
```

The reference warmup uses 128 fixed steps by default. The original model retains
its original optimizer parameter set. Its legacy plain-list attention layers are
saved and restored explicitly, and their train/eval modes are set explicitly.
The same Fourier image-shift operator is used for both methods. These benchmark
controls are recorded rather than presenting historical training logs as a
matched rerun. Perturbation runs default to the large MLP head with a 0.5
axis-angle L2 penalty, selected from historical matched-initialization evidence.
`--pose-model table` selects the indexed differentiable 6D rotation table;
`hybrid` adds both components. Optional per-particle YX translations accompany
the table. Table parameters have their own learning rate, zero weight decay and
a penalty relative to the supplied/HPS poses. Ab-initio runs default to hybrid.

## Evaluate one Output dir or compare methods

```bash
python scripts/benchmark_optpose.py evaluate --outdir results/optpose/full/opt
python scripts/benchmark_optpose.py compare \
  --runs Original=results/optpose/full/original Opt=results/optpose/full/opt \
  --output results/optpose/comparison
```

`evaluate` produces one `learning_curves.png` with all requested angles and three
metric rows. SVG, PDF, CSV and caption files accompany it. `compare` produces one
`method_comparison.png` with RMSD and angular-error grouped bars. Directory labels
are supplied by the caller. Comparisons check particle identities, evaluation
IDs, ground-truth hash, pose mode, angle coverage and endpoint training epochs.

A historical Output dir with `config.py` and `epoch_step/{ckpt.pt,z.npy}` is also
supported:

```bash
python scripts/benchmark_optpose.py evaluate --outdir /path/to/old/atom_1ake \
  --method original --truth-star /path/to/simulation.star \
  --pdb-dir /path/to/pdbs --output results/optpose/historical
```

The source directory stays intact. The converted run is saved under
`results/optpose/historical/imported` and can be passed to `compare`.
Use `--method opt` explicitly for historical optimized runs. Learned pose states
are required: checkpoints that omitted the pose head/cache/table produce an
actionable error and need a rerun. A historical checkpoint represents the
perturbation used during its training; requesting a new perturbation means running
a new experiment. The evaluator does not synthesize missing training curves.

## Recovery without initial poses

To prepare and train the DRGN-AI control from scratch, install DRGN-AI in a
separate compatible environment. The checked reference revision is
`d0872ff68bb65aeaae23f081374f50389dff0a00` of `ml-struct-bio/cryodrgnai`.

```bash
python scripts/benchmark_optpose.py prepare-drgnai --outdir results/drgnai
# Run this next command in the separate DRGN-AI environment:
drgnai train results/drgnai/run --no-analysis
```

The prepared configuration uses particle/CTF-only inputs, heterogeneous
reconstruction, 10,000 random-pose pretraining images, 500,000 HPS images and
20 pose-table epochs. The upstream schedule gives 11 HPS passes on 50,000
particles. The 8 GiB GPU setting uses HPS batch size 1 and refinement batch 8.
The optpose control can reuse the resulting `results/drgnai/run/out`.

```bash
python scripts/benchmark_optpose.py abinit --outdir results/optpose/abinit \
  --drgnai-outdir /path/to/drgnai/run/out
```

This method uses a known reference structure, CTF and particle images. It has a
stronger structural prior than DRGN-AI's learned-volume reconstruction. HPS
initializes poses, followed by indexed pose refinement and structure learning.
The random baseline uses Haar-uniform SO(3) rotations with the recorded seed.
DRGN-AI `pose.N.pkl` files are loaded as trusted local checkpoints. Particle order
is verified using an adjacent `inputs/particles.star` or an explicit
`particle_ids.npy` alongside the checkpoints. The comparison is restricted to
angular errors because DRGN-AI does not provide corresponding Cα structures.

```bash
python scripts/benchmark_optpose.py compare-abinit \
  --outdir results/optpose/abinit/opt/0deg_seed1 \
  --drgnai-outdir /path/to/drgnai/run/out \
  --output results/optpose/abinit/comparison
```

The figure shows actual per-method training passes and final endpoints; a single
ab-initio result is not duplicated into five independent perturbation experiments.
Raw and globally aligned angle errors are stored separately. One right-acting
rotation and one whole-dataset handedness are fit on calibration particles.
The primary ab-initio accuracy gate uses the remaining particles.

## Metric and state contract

- Angular error is `degrees(acos(clip((trace(Rpred @ Rtrue.T)-1)/2,-1,1)))`, including
  in-plane rotation. A view-direction-only angle or Euler-component RMSE is a
  different metric.
- RMSD is computed separately for each particle's predicted and true Cα
  structure, after translation and proper Kabsch rotation alignment, in Å.
- The plotted loss is evaluation image correlation loss, calculated consistently
  across methods; total training loss is stored as `train_loss`.
- Every epoch stores latent codes, predicted rotations/shifts, particle IDs,
  per-particle errors, decoder weights and aggregate metrics. Full task snapshots
  are saved initially, every five epochs and at the endpoint. `resume.pt` includes
  the optimizer, RNG states and all legacy unregistered layers.
- `--resume` requires the same command arguments, inputs and source hashes. It
  resumes at completed epoch boundaries. Completed runs are reused. A free-space
  guard preserves the last completed state if storage approaches capacity.
- Single-seed means have no across-seed confidence claims. Multi-seed endpoint
  plots use sample standard deviations when all compared groups have repeats.

## Validation and diagnostics

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_optpose.py -v
PYTHONPATH=. python tests/test_benchmark_roundtrip.py
python tests/test_hps.py
python scripts/benchmark_optpose.py diagnose --output results/optpose/diagnostics
```

The diagnostic contrasts matched noiseless projections with the supplied noisy
particles, L2 versus normalized-correlation search, and a fixed-reference versus
true-structure oracle control. Oracle structures appear only in the diagnostic.
They are excluded from training and from claims about attainable pose recovery.

Figure source tables and final plot rectangles are always exported. Setting
`CRYODYNA_FIGURE_QA` to a directory containing the publication alignment auditor
enables its additional blocking audit. Public use of the benchmark has no
dependency on a personal skill installation.

## Release gate

The 5°, 10°, 15° and 20° runs must beat the matched original in both RMSD and
rotation error. At 0°, the original already has zero rotation error; numerical
agreement is the angular target. Recovery without initial pose must reach the
stated absolute angular thresholds and beat the DRGN-AI and random controls.
Passing software tests or completing training alone does not establish these
scientific outcomes. Record the measured results and unresolved conditions in
the accompanying acceptance report before committing a release claim.

## 80S and the combined release figure

The fixed-density 80S control, input paths, stage recording and standalone figure
commands are documented in [80S reproduction](80S_REPRODUCTION.md).
Rebuild the committed 80S + 1AKE figure directly from its source tables:

```bash
python scripts/figure_optpose_release.py
```

The reusable trigger is **曲线终点双联图**. Its repository renderer accepts any
dataset label and explicit measured methods, independent of a personal skill:

```bash
python scripts/plot_curve_endpoint.py --csv measured.csv --dataset DATASET \
  --methods Optpose 'Random SO(3)' DRGN-AI --metric-label 'Full SO(3) error (°)' \
  --output results/figure --caption 'Record particles, priors, seeds, gauge and budgets here.'
```

The CSV columns are `method,epoch,value`, optionally `sd` for a consistent
replicate uncertainty definition. Every requested method requires actual values.

The `acceptance` command replays every saved epoch by default. To audit the
scientific endpoint gates with less replay cost, use `acceptance --endpoint-only`;
its JSON explicitly records `prediction_replay_scope: endpoint` and each run
receives `verification_endpoint.json`. The full curves retain the metrics saved
during training. This release snapshot uses endpoint replay for the ten full runs.
