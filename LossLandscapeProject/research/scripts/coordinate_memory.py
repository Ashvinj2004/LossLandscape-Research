"""Coordinate memory and per-unit drift direction.

The same initial FUNCTION is started at three points of its rescaling orbit: first-layer incoming
weights and biases multiplied by c in {1/3, 1, 3} and the second weight matrix divided by c (the
function is identical). Predictions of the two-phase picture:
  * SGD, no label noise: conserves Q to first order -> remembers c (excess stays separated);
  * SGD + label noise: second-order noise drift (law of balance) -> slowly forgets c;
  * Adam: first-order drift -> forgets c within a few epochs.
Every snapshot also stores per-unit Q_j and log(A_j/B_j) at p=1 and p=1/2, so that the direction of
each unit's drift can be correlated with its curvature imbalance offline.

usage: python scripts/coordinate_memory.py [--workers 3]
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

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "coordinate_memory")


def record_epochs(n):
    pts = sorted(set([0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70] + list(range(100, n + 1, 50))))
    return [e for e in pts if e <= n]


def unit_logratio(model, dg, p):
    """Per hidden layer, per unit: log(A_j / B_j) with A, B the summed incoming/outgoing diag(G)^p."""
    import torch
    from lls import rescale as R
    ch = R.chain(model)
    parts = R._split_diag(model, dg)
    out = []
    for l in range(len(ch) - 1):
        DW, Db = parts[l]
        A = (DW.clamp_min(0) ** p).sum(1) + Db.clamp_min(0) ** p
        DW2, _ = parts[l + 1]
        B = torch.zeros_like(A).index_add_(0, ch[l + 1][1], (DW2.clamp_min(0) ** p).sum(0))
        out.append((A.clamp_min(1e-30) / B.clamp_min(1e-30)).log().tolist())
    return out


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
    with torch.no_grad():  # move the initial function along its orbit
        lins = R.linear_layers(model)
        lins[0].weight.mul_(cfg["c"])
        lins[0].bias.mul_(cfg["c"])
        lins[1].weight.div_(cfg["c"])
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"])
    g = torch.Generator().manual_seed(cfg["seed"])
    n, bs, p_noise = len(ytr), cfg["bs"], cfg["label_noise"]
    rec = set(record_epochs(cfg["epochs"]))
    log, t0, step = [], time.time(), 0

    def snapshot(epoch):
        model.eval()
        tl, _ = M.loss_acc(model, xtr, ytr)
        _, te = M.loss_acc(model, xte, yte)
        dg = M.diag_ggn_linear(model, xev)
        ms1, _ = R.orbit_min_trace(model, dg, p=1.0)
        msh, _ = R.orbit_min_trace(model, dg, p=0.5)
        log.append({"epoch": epoch, "step": step, "train_loss": tl, "test_acc": te,
                    "excess1": float(dg.sum()) / ms1, "excess_half": float(dg.clamp_min(0).sqrt().sum()) / msh,
                    "Q": [q.tolist() for q in R.conserved_quantity(model)],
                    "logAB1": unit_logratio(model, dg, 1.0), "logABhalf": unit_logratio(model, dg, 0.5)})
        model.train()

    snapshot(0)
    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx].clone()
            if p_noise > 0:
                flip = torch.rand(len(yb), generator=g) < p_noise
                yb[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if epoch in rec:
            snapshot(epoch)
    return {"cfg": cfg, "log": log, "time": time.time() - t0}


def name(c):
    return f"{c['dataset']}_{c['opt']}_lr{c['lr']}_ln{c['label_noise']}_c{c['c']:.3g}_s{c['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from sweep import keep_awake
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    epochs = {"mnist": 800, "cifar10": 400}
    opts = [("sgd", 0.12, 0.0), ("sgd", 0.12, 0.2), ("adam", 6e-4, 0.0), ("adam", 6e-4, 0.2)]
    cfgs = [dict(dataset=ds, opt=o, lr=lr, label_noise=ln, c=c, seed=s, bs=128, epochs=epochs[ds])
            for ds, (o, lr, ln), c, s in itertools.product(["mnist", "cifar10"], opts, [1 / 3, 1.0, 3.0], [0, 1])]
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
            lg = r["log"]
            print(f"[{i}/{len(todo)}] {name(c)} excess1 {lg[0]['excess1']:.2f}->{lg[3]['excess1']:.2f}->{lg[-1]['excess1']:.2f} "
                  f"loss {lg[-1]['train_loss']:.3f} t={r['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
