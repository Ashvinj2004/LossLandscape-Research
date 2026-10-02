"""Dataset loading from the project's data/ folder (no network needed).

Raw files live in research/data/ (mnist.npz, cifar-10-batches-py/); the Keras download cache is used
as a fallback. Datasets are returned as float32 tensors plus int64 labels and cached as .pt files in
research/.cache/.
"""
import os
import pickle
import numpy as np
import torch

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(RESEARCH_DIR, "data")
CACHE_DIR = os.path.join(RESEARCH_DIR, ".cache")
KERAS_DIR = os.path.expanduser("~/.keras/datasets")


def _find(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("dataset not found (run: python scripts/get_data.py); looked in: " + ", ".join(candidates))


def _load_mnist():
    with np.load(_find(os.path.join(DATA_DIR, "mnist.npz"), os.path.join(KERAS_DIR, "mnist.npz"))) as f:
        xtr, ytr, xte, yte = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    xtr = xtr.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    xte = xte.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def _load_fashion():
    path = os.path.join(KERAS_DIR, "fashion-mnist.npz")
    with np.load(path) as f:
        xtr, ytr, xte, yte = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    xtr = xtr.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    xte = xte.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def _load_cifar10():
    root = _find(os.path.join(DATA_DIR, "cifar-10-batches-py"),
                 os.path.join(KERAS_DIR, "cifar-10-batches-py-target", "cifar-10-batches-py"))

    def read(name):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = read("test_batch")
    xtr = xtr.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    xte = xte.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


LOADERS = {"mnist": _load_mnist, "cifar10": _load_cifar10, "fashion": _load_fashion}


def stratified_subset(y, n, seed):
    """Class-balanced subset of size n (deterministic in seed)."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per = n // len(classes)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in classes])
    rng.shuffle(idx)
    return idx


def load(name, n_train=10000, subset_seed=1234, n_eval=4096, standardize=True):
    """Returns dict with xtr, ytr (training subset), xte, yte (full test set),
    and xev, yev (fixed subset of the training subset used for landscape measures).
    standardize: per-channel zero-mean/unit-std using training-subset statistics."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{name}_n{n_train}_s{subset_seed}_e{n_eval}{'_std' if standardize else ''}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    xtr, ytr, xte, yte = LOADERS[name]()
    if n_train and n_train < len(ytr):
        idx = stratified_subset(ytr, n_train, subset_seed)
        xtr, ytr = xtr[idx], ytr[idx]
    if standardize:
        mu = xtr.mean(axis=(0, 2, 3), keepdims=True)
        sd = xtr.std(axis=(0, 2, 3), keepdims=True)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ev = np.random.RandomState(subset_seed + 1).choice(len(ytr), min(n_eval, len(ytr)), replace=False)
    d = {
        "xtr": torch.from_numpy(xtr), "ytr": torch.from_numpy(ytr),
        "xte": torch.from_numpy(xte), "yte": torch.from_numpy(yte),
        "xev": torch.from_numpy(xtr[ev]), "yev": torch.from_numpy(ytr[ev]),
    }
    torch.save(d, cache)
    return d
