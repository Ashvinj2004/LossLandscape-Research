# Flat in the Wrong Coordinates — code and results

Research code for the paper in [`paper/main.tex`](paper/main.tex): Adam drifts along the
rescaling symmetries of ReLU networks, SGD does not, and this biases flatness comparisons
between optimizers.

Everything runs on a CPU (developed on an i7-13620H laptop, no GPU).

## Layout

```
research/
├── lls/                     library
│   ├── data.py              MNIST / CIFAR-10 from the local Keras cache, subsets, standardization
│   ├── models.py            MLP (in-128-64-10) and a small CNN (no normalization layers)
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
│   ├── gen_correlation.py   sharpness vs generalization (Kendall tau)
│   ├── make_figures.py      all paper figures -> figures/
│   └── test_*.py            correctness checks (diag-GGN vs brute force, rescaling invariances)
├── results/                 one JSON per run (config, training history, checkpoints, measures)
├── figures/                 paper figures (PDF + PNG)
├── paper/                   LaTeX source (compile on Overleaf: main.tex + refs.bib + ../figures)
├── dashboard/               source of the project dashboard page
├── data/                    raw datasets: mnist.npz, cifar-10-batches-py/   (not for git: large)
├── .cache/                  preprocessed tensors, rebuilt automatically      (not for git)
├── checkpoints/             final weights of every main-sweep run            (not for git)
└── requirements.txt         exact package versions
```

## Reproduce

The Python environment lives in `C:\Projects\Loss Landscape Research\.venv`. To rebuild it:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install numpy scipy matplotlib pandas
```

Datasets are read from `research/data/` (copies of the files the original TensorFlow scripts
downloaded); the Keras cache `~/.keras/datasets` is used as a fallback. Run the commands below
from the `research/` folder with that environment's Python.

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
python scripts/make_figures.py
python scripts/fig_landscape.py               # Figure 1 ("same function, different landscape")
python scripts/check_latex.py                 # structural check of paper/main.tex
```

Sweeps are resumable (finished runs are skipped). With 6 worker processes x 2 threads a main
MLP sweep (72 runs) takes about 80 minutes on the laptop above.

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
