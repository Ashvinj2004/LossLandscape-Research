# Flat in the Wrong Coordinates — code and results

Research code for the paper in [`paper/main.tex`](paper/main.tex): Adam drifts along the
rescaling symmetries of ReLU networks, SGD does not, and this biases flatness comparisons
between optimizers.

Everything runs on a CPU. Most experiments ran on an i7-13620H laptop; the coordinate-memory, depth
and Adam-teleport experiments ran on a 4-core cloud machine. No GPU is needed.

## Layout

```
research/
├── lls/                     library
│   ├── data.py              MNIST / CIFAR-10 from research/data/, subsets, standardization
│   ├── models.py            MLP (in-128-64-10), depth-sweep MLPs, BatchNorm MLP, small CNN
│   ├── train.py             training with loss-matched checkpoints; SGD, SGD+M, SAM, Adam, Adam-Q
│   ├── measures.py          17 sharpness measures, exact diagonal Gauss-Newton, Lanczos, PGD
│   ├── rescale.py           rescaling orbits: balancing, minimum sharpness, drift removal (Adam-Q)
│   └── analysis.py          sweep JSON -> DataFrame
├── scripts/
│   ├── sweep.py             all experiment grids (calibration, main, intervention, replication, CNN)
│   ├── drift_experiment.py  exact per-step drift decomposition (mechanism, scaling law)
│   ├── analyze_main.py      tables: P(Adam flatter), orbit excess, imbalance
│   ├── analyze_bn.py        BatchNorm sweep: raw vs per-weight-invariant measures, pre-BN norms
│   ├── orbit_dynamics.py    long-horizon runs with/without label noise (two-phase picture)
│   ├── coordinate_memory.py same function started at three orbit points (+ analyze_coordinate_memory.py)
│   ├── analyze_depth.py     depth sweep: function / coordinate split per depth
│   ├── gen_correlation.py   sharpness vs generalization (Kendall tau)
│   ├── make_figures.py      all paper figures -> figures/
│   └── test_*.py            correctness checks (diag-GGN vs brute force, rescaling invariances)
├── results/                 one JSON per run (config, training history, checkpoints, measures)
├── figures/                 paper figures (PDF + PNG)
├── paper/                   main.tex, refs.bib, compiled main.pdf (figures load from ../figures)
├── dashboard/               source of the project dashboard page
├── data/                    raw datasets: mnist.npz, cifar-10-batches-py/   (not for git: large)
├── .cache/                  preprocessed tensors, rebuilt automatically      (not for git)
├── checkpoints/             final weights of every main-sweep run            (not for git)
└── requirements.txt         exact package versions
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or plain `pip install torch`
pip install -r requirements.txt
python scripts/get_data.py
```

`get_data.py` downloads MNIST and CIFAR-10 into `research/data/` and verifies their MD5 checksums.
If the CIFAR-10 host is unreachable, it rebuilds the dataset from fast.ai's lossless PNG copy. That
copy has a different order, so it gives a different 10k training subset, and it is marked by a
`REORDERED` file. The Keras cache `~/.keras/datasets` is used as a fallback. Run the commands below
from the `research/` folder.

Every figure and table can be regenerated from the stored JSONs without retraining:
`python scripts/make_figures.py` and the `analyze_*.py` scripts. The full pipeline:

```bash
python scripts/test_ggn.py             # diag-GGN matches brute force
python scripts/test_rescale.py         # function preserved, transport exact, orbit-min is a minimum
python scripts/test_invariance.py      # which perturbation measures are invariant to which rescalings
python scripts/sweep.py calib_cifar10  # learning-rate calibration (training only)
python scripts/sweep.py main_cifar10_mlp
python scripts/sweep.py main_mnist_mlp
python scripts/sweep.py interv_cifar10 # Adam-Q intervention
python scripts/sweep.py interv_mnist
python scripts/sweep.py orig_replication
python scripts/sweep.py cnn_cifar10
python scripts/drift_experiment.py            # SGD, SGD+M, SAM, Adam
python scripts/drift_experiment.py --extra    # RMSprop, signSGD
python scripts/analyze_main.py main_cifar10_mlp main_mnist_mlp
python scripts/analyze_intervention.py main_cifar10_mlp interv_cifar10 main_mnist_mlp interv_mnist
python scripts/bootstrap_ci.py main_cifar10_mlp main_mnist_mlp
python scripts/gen_correlation.py main_cifar10_mlp main_mnist_mlp
python scripts/sweep.py func_adamq           # Adam-Q controls
python scripts/sweep.py func_teleport        # symmetry teleportation of SGD / SGD+M
python scripts/sweep.py bn_cifar10           # BatchNorm MLP + weight decay
python scripts/analyze_bn.py --json results/bn_cifar10_summary.json
python scripts/orbit_dynamics.py
python scripts/coordinate_memory.py
python scripts/analyze_coordinate_memory.py --json results/coordinate_memory_summary.json
python scripts/sweep.py depth_cifar10        # depth 1-8 MLPs
python scripts/analyze_depth.py --json results/depth_cifar10_summary.json
python scripts/sweep.py adam_orbit           # Adam held at the min-trace or min-norm orbit point
python scripts/make_figures.py
python scripts/fig_landscape.py               # Figure 1 ("same function, different landscape")
python scripts/check_latex.py                 # structural check of paper/main.tex
```

Sweeps are resumable (finished runs are skipped). With 6 worker processes x 2 threads a main
MLP sweep (72 runs) takes about 80 minutes on the laptop above.

## Experiment index

| results/ folder | runs | what it tests | paper |
|---|---:|---|---|
| `pilot_cifar` | 12 | first look; edge-of-stability observation | App. E |
| `calib_cifar10`, `calib_mnist` | 96 | learning-rate calibration (training only) | App. D |
| `main_cifar10_mlp`, `main_mnist_mlp` | 144 | loss-matched audit: SGD, SGD+M, SAM, Adam, 17 measures | §5.2, §5.6 |
| `drift` | 90 | exact per-step drift decomposition, scaling law, RMSprop, signSGD | §5.1, App. B |
| `interv_cifar10`, `interv_mnist` | 36 | Adam-Q causal intervention | §5.3 |
| `func_adamq` | 36 | Adam-Q controls (every 50 steps, one layer, no moment transform) | §5.3 |
| `func_teleport` | 32 | teleporting SGD / SGD+M to the minimum-trace point | §5.4 |
| `orbit_dynamics` | 32 | 400-600 epochs with / without label noise | §5.5 |
| `coordinate_memory` | 48 | same function started at three orbit points | §5.5 |
| `orig_replication` | 20 | the original study's protocol, 5 seeds | §5.7 |
| `cnn_cifar10` | 18 | CNN audit (channel rescaling) | §5.8 |
| `depth_cifar10` | 36 | MLPs with 1-8 hidden layers | §5.9 |
| `bn_cifar10` | 24 | BatchNorm MLP, weight decay, AdamW | §5.10 |
| `adam_orbit` | 12 | Adam held at the min-trace vs the min-norm orbit point | §5.3 |

Each run's JSON holds its configuration, training curve, and every measurement at every
loss-matched checkpoint. `*_summary.json` files hold the numbers quoted in the paper.

## Correctness checks that were run

| check | result |
|---|---|
| fast diag-GGN (MLP closed form) vs per-sample `vmap` | max abs error 2e-8 |
| `vmap` diag-GGN vs brute-force J^T S J | max abs error 3e-8 |
| function preserved by balancing / orbit-min rescaling (MLP, CNN) | max output change 3e-6 / 1e-6 |
| analytic transport of diag(G) along the orbit vs recomputation | relative error 2e-7 (MLP), 3e-7 (CNN) |
| orbit-min trace vs raw, min-norm and random rescalings | always the smallest |
| per-step drift decomposition closes (Eq. 1 of the paper) | max error 1.3e-6 |
| SGD first-order drift (must be exactly 0) | < 3e-7 (float32 round-off) |
| multiplicative, ASAM, Σ w²H_ii under random neuron / BN rescalings (`test_invariance.py`) | unchanged (< 2e-4 relative) |
| filter-normalized sharpness under the same rescalings | changes ×9.5 (neuron), ×19 (BN affine): only row-scale invariant |
