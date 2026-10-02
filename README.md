# Flat in the Wrong Coordinates

**Optimizer-induced symmetry drift biases sharpness comparisons between Adam and SGD.**

Ashvin J — School of Computer Science and Engineering, VIT-AP University

This repository holds the paper, all code, and the raw result of every training run behind it
(about 640 networks, all trained on CPUs).

- Paper: [`LossLandscapeProject/research/paper/main.pdf`](LossLandscapeProject/research/paper/main.pdf)
  (LaTeX source: `main.tex`, `refs.bib`)
- Code and detailed reproduction guide: [`LossLandscapeProject/research/`](LossLandscapeProject/research/README.md)

---

## The question

Do Adam and SGD find minima of different flatness? Many papers measure this, and their answers
disagree. A first version of this project measured it with a standard tool, the relative loss increase
under random weight noise, and found Adam's minima *flatter* than SGD's on MNIST and CIFAR-10. This
repository shows why that result, and comparisons like it, can be an artifact.

## The answer in brief

A ReLU network computes exactly the same function if one neuron's incoming weights are multiplied by
any `c > 0` and its outgoing weights divided by `c`. Most sharpness measures (Hessian trace, random
perturbations, SAM sharpness) change under this rescaling. So they measure two things at once: how
sharp the learned **function** is, and which of the equivalent **coordinates** the network sits at.
For the Hessian trace the split is exact:

```
raw trace  =  minimum sharpness (function only)  ×  orbit excess (coordinates only, ≥ 1)
```

The key observation is that **the optimizer chooses the coordinates**:

- **SGD cannot move between equivalent coordinates.** Its first-order drift along these symmetries
  is exactly zero (proof in the paper; measured below 3×10⁻⁷).
- **Adam can, and does.** It drifts at a rate proportional to η·√fan-in and settles near the
  coordinates where its trace is smallest. RMSprop and sign descent behave the same way.

Raw sharpness comparisons between Adam and SGD therefore carry a coordinate factor chosen by the
optimizer, and that factor is as large as the effects being measured.

## Main results

| Finding | Evidence |
|---|---|
| Adam keeps its orbit excess at ≈1.2; SGD, SGD+momentum and SAM sit at 1.8–3.4 | 144 loss-matched MLP runs, 17 measures |
| On a CIFAR-10 MLP, the coordinate factor favours Adam by ≈2× and **reverses** the ranking: P(Adam rated flatter than SGD) = 0.62–0.75 by raw trace-type measures, 0.01–0.22 by minimum sharpness | Table 1, Fig. 3 |
| On MNIST it **flips** the raw ranking during training; on a CNN it **masks** a 2× real difference | §5.2, §5.8 |
| **Causal test.** Removing Adam's drift after every step (Adam-Q, function unchanged) raises raw trace ×1.91 and leaves minimum sharpness unchanged (×0.98) | 36 + 36 runs |
| The bias hurts **cross-optimizer** sharpness–generalization correlations (Kendall τ 0.38 → 0.61 when it is removed) and leaves within-optimizer ones intact | §5.6 |
| **Two phases.** A fast first-order drift (Adam reaches excess ≈1.1 within 5 epochs), then a slow noise-driven drift. With label noise, SGD moves toward minimum trace and Adam toward minimum Σ√G_ii, the destinations predicted by implicit-regularization theory, in **20/20 runs each** | §5.1, §5.5; 80 runs |
| The excess is set by training dynamics, not by the starting point: the same function started at three orbit points ends at nearly the same excess, and Adam ends below SGD from every start | §5.5; 48 runs |
| Moving SGD to Adam's preferred coordinates (symmetry teleportation) **destabilizes** it: 10/16 teleported runs diverge, against 0/16 without teleports, because each teleport raises λ_max toward 2/η | §5.4, 32 runs |
| The coordinate factor does **not** grow with depth (×1.4–1.8 at 1–8 hidden layers), while Adam's real sharpness does: the bias reverses the ranking in shallow MLPs and masks it in deep ones | §5.9, 36 runs |
| With **BatchNorm** the bias points the other way: raw measures track learning-rate-dependent weight norms and rate Adam sharper; per-weight-invariant measures rate it flatter in all 8 learning-rate pairings | §5.10, 24 runs |
| The original "Adam is flatter" result replicates, but it is the coordinate factor on CIFAR-10 and a training-loss mismatch on MNIST | 20 runs, 5 seeds |

We do **not** claim that Adam's minima are always sharper: invariant measures rate Adam sharper on
plain MLPs and the CNN, but flatter with BatchNorm. The claim is that raw-coordinate comparisons
contain an optimizer-dependent bias with a known mechanism, large enough to change conclusions.

**Recommendations:**
1. Compare optimizers at matched training loss.
2. Report a rescaling-invariant measure (minimum sharpness, multiplicative or ASAM sharpness, or
   Σ w²·H_ii), or at least the orbit excess.
3. Don't rely on min-norm canonicalization or filter normalization alone. Filter normalization is
   invariant to row scaling only, and a neuron rescaling changes it by up to 19× in our tests.

## Repository layout

```
LossLandscapeProject/
├── research/            the study (start here)
│   ├── paper/           main.tex, refs.bib, main.pdf
│   ├── lls/             PyTorch library: data, models, training, measures, rescaling maths
│   ├── scripts/         sweeps, analyses, figures, correctness tests
│   ├── results/         one JSON per training run + analysis summaries + logs
│   ├── figures/         all paper figures (PDF + PNG)
│   └── dashboard/       source of the project's summary web page
├── method*.py           the original TensorFlow scripts this project started from
├── plots/, results/     outputs of those original scripts
```

## Quick start

CPU only; Python 3.10+.

```bash
cd LossLandscapeProject/research
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or: pip install torch
pip install -r requirements.txt
python scripts/get_data.py          # MNIST + CIFAR-10 (checksum-verified)
python scripts/test_rescale.py      # correctness checks
python scripts/test_invariance.py
python scripts/make_figures.py      # all figures from the stored results (no training needed)
```

Every figure and number in the paper can be regenerated from `results/` without retraining. To rerun
an experiment, use `python scripts/sweep.py <name>`. Sweeps are resumable: a run whose result JSON
exists is skipped. The full list of commands, and which paper section each one feeds, is in
[`LossLandscapeProject/research/README.md`](LossLandscapeProject/research/README.md).

## Data note

MNIST and CIFAR-10 come from their standard sources, and `get_data.py` verifies MD5 checksums. When
the CIFAR-10 host (cs.toronto.edu) is unreachable, the script rebuilds the dataset from fast.ai's
lossless PNG copy. The images are identical but stored in a different order, so the 10k training
subset differs. The coordinate-memory, depth and Adam-teleport experiments were run this way; the
paper says so where relevant.

## Citation

```bibtex
@misc{ashvin2026flat,
  title  = {Flat in the Wrong Coordinates: Optimizer-Induced Symmetry Drift Biases
            Sharpness Comparisons Between Adam and SGD},
  author = {Ashvin J},
  year   = {2026},
  note   = {https://github.com/Ashvinj2004/LossLandscape-Research}
}
```
