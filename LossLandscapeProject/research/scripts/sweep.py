"""Parallel sweep runner. Each run writes one JSON to results/<sweep>/<run_name>.json (resumable).

usage: python scripts/sweep.py <sweep_name> [--workers 7] [--threads 2]
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls.train import run, run_name  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

LR_GRID = {
    "sgd": [0.01, 0.03, 0.1, 0.3],
    "sgdm": [0.001, 0.003, 0.01, 0.03],
    "sam": [0.001, 0.003, 0.01, 0.03],
    "adam": [1e-4, 3e-4, 1e-3, 3e-3],
}


def grid(sweep):
    cfgs = []
    if sweep == "pilot_cifar":
        for opt in ["sgd", "sgdm", "adam"]:
            for lr in LR_GRID[opt]:
                cfgs.append(dict(dataset="cifar10", arch="mlp", opt=opt, lr=lr, bs=128, seed=0,
                                 max_epochs=300, level="cheap", save_model=False))
    elif sweep.startswith("calib_"):
        # calib_<dataset>: training-only (no measures) to choose lr grids that reach the loss targets
        ds = sweep.split("_")[1]
        for opt in ["sgd", "sgdm", "adam", "sam"]:
            for lr in CALIB_LRS[opt]:
                for bs in [32, 128, 512]:
                    cfgs.append(dict(dataset=ds, arch="mlp", opt=opt, lr=lr, bs=bs, seed=0, max_epochs=200,
                                     level="none", save_model=False, targets=[1.0, 0.3, 0.1, 0.03, 0.01]))
    elif sweep.startswith("main_"):
        # main_<dataset>_<arch>
        # lr = base * sqrt(bs / 32) (square-root scaling): every combination trains stably (see calib_*).
        _, ds, arch = sweep.split("_")
        for seed, opt, bs in itertools.product([0, 1, 2], ["sgd", "sgdm", "adam", "sam"], [32, 128, 512]):
            for base in BASE_LRS[opt]:
                lr = float(f"{base * (bs / 32) ** 0.5:.6g}")
                cfgs.append(dict(dataset=ds, arch=arch, opt=opt, lr=lr, bs=bs, seed=seed, base_lr=base,
                                 max_epochs=MAX_EPOCHS.get((ds, arch), 300), level="full",
                                 targets=[1.0, 0.3, 0.1, 0.03, 0.01]))
    elif sweep.startswith("interv_"):
        # interv_<dataset>: Adam-Q (Adam + orbit-drift removal), same grid as the main sweep's Adam runs
        ds = sweep.split("_")[1]
        for seed, bs in itertools.product([0, 1, 2], [32, 128, 512]):
            for base in BASE_LRS["adam"]:
                lr = float(f"{base * (bs / 32) ** 0.5:.6g}")
                cfgs.append(dict(dataset=ds, arch="mlp", opt="adamq", lr=lr, bs=bs, seed=seed, base_lr=base,
                                 max_epochs=300, level="full", targets=[1.0, 0.3, 0.1, 0.03, 0.01]))
    elif sweep == "func_adamq":
        # Does Adam's drift help it train? Controls for the Adam-Q stalls (training only).
        variants = [dict(opt="adam"), dict(opt="adamq"), dict(opt="adamq", q_transform=False),
                    dict(opt="adamq", q_every=50), dict(opt="adamq", q_layers=[0]), dict(opt="adamq", q_layers=[1])]
        for seed, (lr, bs), v in itertools.product([0, 1], [(3e-4, 32), (6e-4, 128), (1.2e-3, 512)], variants):
            cfgs.append(dict(dataset="cifar10", arch="mlp", lr=lr, bs=bs, seed=seed, max_epochs=300, level="none",
                             save_model=False, targets=[1.0, 0.3, 0.1, 0.03, 0.01], **v))
    elif sweep == "func_teleport":
        # Does moving SGD to its minimum-sharpness coordinates help it train? (training only)
        for seed, tele in itertools.product([0, 1], [0, 100]):
            for opt, lrs in (("sgd", [0.06, 0.12, 0.24, 0.48]), ("sgdm", [0.006, 0.012, 0.024, 0.048])):
                for lr in lrs:
                    c = dict(dataset="cifar10", arch="mlp", opt=opt, lr=lr, bs=128, seed=seed, max_epochs=200,
                             level="none", save_model=False, targets=[1.0, 0.3, 0.1, 0.03, 0.01])
                    if tele:
                        c["teleport_every"] = tele
                    cfgs.append(c)
    elif sweep == "bn_cifar10":
        # Normalization layers + weight decay: MLP with BatchNorm, measured with batch statistics.
        configs = [("sgd", 0.1, 0.0), ("sgd", 0.3, 0.0), ("sgdm", 0.01, 0.0), ("sgdm", 0.03, 0.0),
                   ("adam", 3e-4, 0.0), ("adam", 1e-3, 0.0), ("sgdm", 0.03, 5e-4), ("adamw", 1e-3, 0.05)]
        for seed, (opt, lr, wd) in itertools.product([0, 1, 2], configs):
            c = dict(dataset="cifar10", arch="mlpbn", opt=opt, lr=lr, bs=128, seed=seed, max_epochs=150,
                     level="full", targets=[1.0, 0.3, 0.1])
            if wd:
                c["wd"] = wd
            cfgs.append(c)
    elif sweep == "orig_replication":
        # The original study's protocol: raw [0,1] inputs, full training set, 20 epochs, bs 64,
        # SGD lr 0.01 vs Adam lr 0.001, measured once at the end (fixed epochs, NOT loss-matched).
        for ds, seed, (opt, lr) in itertools.product(["mnist", "cifar10"], range(5), [("sgd", 0.01), ("adam", 0.001)]):
            cfgs.append(dict(dataset=ds, arch="mlp", opt=opt, lr=lr, bs=64, seed=seed, n_train=None,
                             standardize=False, max_epochs=20, targets=[], level="full"))
    elif sweep == "cnn_cifar10":
        for seed, (opt, lrs) in itertools.product([0, 1, 2], CNN_LRS.items()):
            for lr in lrs:
                cfgs.append(dict(dataset="cifar10", arch="cnn", opt=opt, lr=lr, bs=128, seed=seed, n_eval=1024,
                                 max_epochs=150, level="lean", targets=[1.0, 0.1, 0.01], time_limit=1e9))
    elif sweep == "adam_orbit":
        # Why does Adam-Q stall? Hold Adam at a chosen orbit point by teleporting every 100 steps
        # (function-preserving, Adam moments transformed covariantly): to the minimum-trace point,
        # or to the minimum-norm (balanced) point, where Adam-Q ends up. Same configs as func_adamq.
        for seed, (lr, bs), to in itertools.product([0, 1], [(3e-4, 32), (6e-4, 128), (1.2e-3, 512)],
                                                    ["min", "balanced"]):
            cfgs.append(dict(dataset="cifar10", arch="mlp", opt="adam", lr=lr, bs=bs, seed=seed,
                             max_epochs=300, level="lean", targets=[1.0, 0.3, 0.1, 0.03, 0.01],
                             teleport_every=100, teleport_to=to, save_model=False, time_limit=1e9))
    elif sweep == "depth_cifar10":
        # Does the coordinate factor grow with depth? Uniform-width MLPs, He init, bs 128, the main sweep's
        # larger base rate per optimizer (square-root scaled to bs 128).
        for seed, L, (opt, lr) in itertools.product([0, 1, 2], [1, 2, 4, 8],
                                                     [("sgd", 0.06), ("sgdm", 0.006), ("adam", 6e-4)]):
            cfgs.append(dict(dataset="cifar10", arch=f"mlp_d{L}_w128", opt=opt, lr=lr, bs=128, seed=seed,
                             max_epochs=300, level="lean", targets=[1.0, 0.3, 0.1], save_model=False,
                             time_limit=1e9))
    else:
        raise ValueError(sweep)
    return cfgs


CALIB_LRS = {
    "sgd": [0.01, 0.03, 0.1, 0.3],
    "sgdm": [0.001, 0.003, 0.01, 0.03],
    "sam": [0.001, 0.003, 0.01, 0.03],
    "adam": [3e-5, 1e-4, 3e-4, 1e-3],
}
BASE_LRS = {"sgd": [0.01, 0.03], "sgdm": [0.001, 0.003], "sam": [0.001, 0.003], "adam": [1e-4, 3e-4]}
MAX_EPOCHS = {}
CNN_LRS = {"sgd": [0.03, 0.1], "sgdm": [0.003, 0.01], "adam": [1e-4, 3e-4]}


def _work(cfg, out_path):
    try:
        r = run(cfg)
        with open(out_path, "w") as f:
            json.dump(r, f)
        return out_path, r["status"], r["time"]
    except Exception:
        return out_path, "error:" + traceback.format_exc()[-800:], 0.0


def keep_awake():
    """Ask Windows not to idle-sleep while this process runs (released automatically on exit).
    Does not change any setting and does not block a user-initiated sleep or lid close."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out_dir = os.path.join(RESULTS, a.sweep)
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for cfg in grid(a.sweep):
        cfg["threads"] = a.threads
        p = os.path.join(out_dir, run_name(cfg) + ".json")
        if not os.path.exists(p):
            todo.append((cfg, p))
    print(f"{a.sweep}: {len(todo)} runs to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_work, c, p) for c, p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            p, st, t = f.result()
            print(f"[{i}/{len(todo)}] {os.path.basename(p)} {st} {t:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
