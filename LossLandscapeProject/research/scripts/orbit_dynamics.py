"""Long-horizon orbit dynamics, with and without label noise.

Prediction of the unified picture (paper, Sec. 5.1): every optimizer drifts toward curvature-balanced
(minimum-sharpness) coordinates, SGD at O(eta^2/B) per step (law of balance; label-noise SGD minimizes
tr H) and Adam at O(eta). With enough noise-driven time SGD's orbit excess should therefore fall toward 1.
Li, Wen & Lyu (2025) instead predict that label-noise Adam balances sum_i sqrt(H_ii) (p = 1/2).

Records, every few epochs: clean train loss, test acc, tr G, MS_1, MS_1/2, orbit excess for p = 1 and
p = 1/2, and the per-unit curvature balance |log(A_j/B_j)| at p = 1 and p = 1/2.

usage: python scripts/orbit_dynamics.py [--workers 6]
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "orbit_dynamics")


def record_epochs(n):
    pts = sorted(set([0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70] + list(range(100, n + 1, 25))))
    return [e for e in pts if e <= n]


def one(cfg):
    import torch
    import torch.nn.functional as F
    from lls import data as D, measures as M, rescale as R
    from lls.models import MLP
    from lls.train import make_optimizer

    torch.set_num_threads(2)
    d = D.load(cfg["dataset"])
    xtr, ytr, xte, yte, xev = d["xtr"], d["ytr"], d["xte"], d["yte"], d["xev"]
    torch.manual_seed(cfg["seed"])
    model = MLP(xtr[0].numel())
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"])
    g = torch.Generator().manual_seed(cfg["seed"])
    n, bs, p_noise = len(ytr), cfg["bs"], cfg["label_noise"]
    rec = set(record_epochs(cfg["epochs"]))
    log, t0, step = [], time.time(), 0

    def snapshot(epoch):
        model.eval()
        tl, ta = M.loss_acc(model, xtr, ytr)
        _, te = M.loss_acc(model, xte, yte)
        dg = M.diag_ggn_linear(model, xev)
        ms1, _ = R.orbit_min_trace(model, dg, p=1.0)
        msh, _ = R.orbit_min_trace(model, dg, p=0.5)
        tr = float(dg.sum())
        sq = float(dg.clamp_min(0).sqrt().sum())
        log.append({"epoch": epoch, "step": step, "train_loss": tl, "train_acc": ta, "test_acc": te,
                    "trG": tr, "MS1": ms1, "sqrtG": sq, "MShalf": msh, "excess1": tr / ms1, "excess_half": sq / msh,
                    "balance1": R.curvature_balance(model, dg, 1.0), "balance_half": R.curvature_balance(model, dg, 0.5),
                    "imbalance": R.imbalance(model), "w_norm": float(M.get_flat(model).norm()),
                    "time": time.time() - t0})
        model.train()

    snapshot(0)
    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx].clone()
            if p_noise > 0:  # fresh label noise every step (Blanc et al. 2020; Damian et al. 2021)
                flip = torch.rand(len(yb), generator=g) < p_noise
                yb[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if epoch in rec:
            snapshot(epoch)
            if not (log[-1]["train_loss"] < 50):
                break
    return {"cfg": cfg, "log": log, "time": time.time() - t0}


def name(c):
    return f"{c['dataset']}_{c['opt']}_lr{c['lr']}_ln{c['label_noise']}_s{c['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from sweep import keep_awake
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    lrs = {"sgd": [0.06, 0.12], "adam": [2e-4, 6e-4]}
    epochs = {"mnist": 600, "cifar10": 400}
    cfgs = []
    for ds, seed, ln in itertools.product(["mnist", "cifar10"], [0, 1], [0.0, 0.2]):
        for opt, lr_list in lrs.items():
            for lr in lr_list:
                cfgs.append(dict(dataset=ds, opt=opt, lr=lr, bs=128, seed=seed, label_noise=ln, epochs=epochs[ds]))
    todo = [c for c in cfgs if not os.path.exists(os.path.join(OUT, name(c) + ".json"))]
    print(len(todo), "runs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print("ERROR", name(c), repr(e), flush=True)
                continue
            with open(os.path.join(OUT, name(c) + ".json"), "w") as fh:
                json.dump(r, fh)
            first, last = r["log"][1], r["log"][-1]
            print(f"[{i}/{len(todo)}] {name(c)} ep{last['epoch']} loss {last['train_loss']:.3f} "
                  f"excess1 {first['excess1']:.2f}->{last['excess1']:.2f} excess_half {first['excess_half']:.2f}->"
                  f"{last['excess_half']:.2f} t={r['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
