# Loss Landscape Research

**Flat in the Wrong Coordinates: optimizer-induced symmetry drift biases sharpness comparisons
between Adam and SGD.** Paper draft, code and all raw results.

- `LossLandscapeProject/research/` — the study: PyTorch library (`lls/`), experiment scripts,
  per-run results (`results/`), figures, LaTeX paper (`paper/`), project dashboard source.
  See `LossLandscapeProject/research/README.md` for details.
- `LossLandscapeProject/method*.py`, `plots/`, `results/` — the original TensorFlow study that
  this work started from.

## Quick start (Linux / macOS / cloud)

```bash
cd LossLandscapeProject/research
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/get_data.py          # MNIST + CIFAR-10, checksum-verified
python scripts/test_rescale.py      # correctness checks
```

Experiments are CPU-only and resumable: `python scripts/sweep.py <name>` skips runs whose result
JSON already exists.
