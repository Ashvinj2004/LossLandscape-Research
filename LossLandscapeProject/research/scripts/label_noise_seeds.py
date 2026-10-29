"""More seeds for the label-noise runs of Sec. 5.5, with a frozen-curvature test of the orbit motion.

Same setting as orbit_dynamics.py at the learning rates shared with coordinate_memory.py (SGD 0.12,
Adam 6e-4, batch 128, label noise 0.2; MNIST 600 and CIFAR-10 400 epochs), for seeds 2-7.

The orbit excess changes both when the parameters move along the rescaling orbit and when the curvature
diag(G) itself changes. To separate the two, between consecutive snapshots t < t' (epochs >= 100) we
keep diag(G) at t fixed, move the parameters at t along their own orbit until every unit's Q_j equals
its value at t' (the orbit component of the motion), and record how the frozen objectives sum_i G_ii
(p = 1) and sum_i sqrt(G_ii) (p = 1/2) change. Attraction toward the minimum-trace point predicts that
the orbit motion lowers the frozen p = 1 objective by more than the p = 1/2 one; attraction toward the
minimum of sum sqrt(G) predicts the opposite.

usage: python scripts/label_noise_seeds.py [--workers 2]
"""
import argparse
import copy
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "label_noise_seeds")


def record_epochs(n):
    pts = sorted(set([0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70] + list(range(100, n + 1, 25))))
    return [e for e in pts if e <= n]


def move_along_orbit(model, q_target, sweeps=30):
    """Rescale `model` in place so that every unit's Q_j matches q_target; return the cumulative
    per-layer unit scales (incoming weights multiplied by c, outgoing divided by c)."""
    import torch
    from lls import rescale as R
    ch = R.chain(model)
    scales = [torch.ones(R._n_units(m)) for m, _ in ch[:-1]]
    for _ in range(sweeps):
        for l in range(len(ch) - 1):
            n_in, n_out = R._in_out_norm2(ch, l)
            q = q_target[l]
            x = (q + torch.sqrt(q * q + 4 * n_in * n_out)) / (2 * n_in.clamp_min(1e-30))
            c = x.clamp_min(1e-30).sqrt()
            R._scale_unit(ch, l, c)
            scales[l] = scales[l] * c
    return scales


def one(cfg):
    import numpy as np
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
    log, frozen, t0, step = [], [], time.time(), 0
    prev = None  # (epoch, model copy, diag G) of the previous snapshot at epoch >= 100

    def snapshot(epoch):
        nonlocal prev
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
                    "time": time.time() - t0})
        if epoch >= 100:
            if prev is not None:
                e0, m0, dg0 = prev
                q_now = [q.clone() for q in R.conserved_quantity(model)]
                m1 = copy.deepcopy(m0)
                scales = move_along_orbit(m1, q_now)
                resid = max(float(((a - b).abs().max() / b.abs().max())) for a, b in zip(R.conserved_quantity(m1), q_now))
                dgt = R.transport_diag(m0, dg0, scales)
                f = lambda v, p: float(v.double().clamp_min(0).pow(p).sum())
                frozen.append({"from": e0, "to": epoch, "q_residual": resid,
                               "dlog_frozen1": float(np.log(f(dgt, 1.0) / f(dg0, 1.0))),
                               "dlog_frozen_half": float(np.log(f(dgt, 0.5) / f(dg0, 0.5)))})
            prev = (epoch, copy.deepcopy(model), dg.clone())
        model.train()

    snapshot(0)
    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx].clone()
            if p_noise > 0:  # fresh label noise every step, as in orbit_dynamics.py
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
    return {"cfg": cfg, "log": log, "frozen": frozen, "time": time.time() - t0}


def name(c):
    return f"{c['dataset']}_{c['opt']}_lr{c['lr']}_ln{c['label_noise']}_s{c['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default="2,3,4,5,6,7")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lrs = {"sgd": 0.12, "adam": 6e-4}
    epochs = {"mnist": 600, "cifar10": 400}
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = [dict(dataset=ds, opt=o, lr=lrs[o], bs=128, seed=s, label_noise=0.2, epochs=epochs[ds])
            for ds, s, o in itertools.product(["mnist", "cifar10"], seeds, ["sgd", "adam"])]
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
            last = r["log"][-1]
            fz = r["frozen"]
            print(f"[{i}/{len(todo)}] {name(c)} ep{last['epoch']} loss {last['train_loss']:.3f} "
                  f"excess1 {last['excess1']:.2f} excess_half {last['excess_half']:.2f} "
                  f"frozen sum dlog p=1 {sum(x['dlog_frozen1'] for x in fz):+.3f} "
                  f"p=1/2 {sum(x['dlog_frozen_half'] for x in fz):+.3f} t={r['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
