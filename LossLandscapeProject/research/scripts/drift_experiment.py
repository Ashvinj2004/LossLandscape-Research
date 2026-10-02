"""Mechanism experiment: exact per-step decomposition of the drift of the rescaling 'conserved quantity'
    Q_j = ||w_in,j||^2 - ||w_out,j||^2
for every hidden unit j. For a step u, ||w+u||^2 - ||w||^2 = 2<w,u> + ||u||^2 exactly, so
    dQ_j = [2<w_in,u_in> - 2<w_out,u_out>]   (first order; == 0 for SGD by the Euler identity)
         + [||u_in||^2 - ||u_out||^2]        (second order).
We vary the optimizer, the learning rate and the input dimension d_in (CIFAR-10 average-pooled to
8x8, 16x16, 32x32) and record cumulative first/second-order drift, imbalance and the balanced/raw
sharpness ratio at the end.

usage: python scripts/drift_experiment.py [--workers 4]
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "drift")


def one(cfg):
    import torch
    import torch.nn.functional as F
    from lls import data as D, measures as M, rescale as R
    from lls.models import MLP
    from lls.train import make_optimizer

    torch.set_num_threads(2)
    d = D.load("cifar10")
    xtr, ytr = d["xtr"], d["ytr"]
    xev = d["xev"]
    if cfg["res"] < 32:
        k = 32 // cfg["res"]
        xtr, xev = F.avg_pool2d(xtr, k), F.avg_pool2d(xev, k)
    torch.manual_seed(cfg["seed"])
    model = MLP(xtr[0].numel())
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"])
    ch = R.chain(model)
    nL = len(ch) - 1
    cum1 = [torch.zeros(R._n_units(ch[l][0]), dtype=torch.float64) for l in range(nL)]
    cum2 = [torch.zeros_like(c) for c in cum1]
    q0 = [q.double().clone() for q in R.conserved_quantity(model)]
    g = torch.Generator().manual_seed(cfg["seed"])
    n, bs, rho = len(ytr), cfg["bs"], 0.05
    log, step = [], 0
    t0 = time.time()

    def unit_terms(w_prev, w_new):
        """per-unit (first-order, second-order) contributions to dQ for each hidden layer."""
        res = []
        for l in range(nL):
            (Wa, ba), (Wb, _) = w_prev[l], w_prev[l + 1]
            (Wa2, ba2), (Wb2, _) = w_new[l], w_new[l + 1]
            ua, uba, ub = Wa2 - Wa, ba2 - ba, Wb2 - Wb
            gidx = ch[l + 1][1]
            in1 = 2 * ((Wa * ua).sum(1) + ba * uba)
            in2 = (ua * ua).sum(1) + uba * uba
            out1 = torch.zeros(len(in1), dtype=Wa.dtype).index_add_(0, gidx, 2 * (Wb * ub).sum(0))
            out2 = torch.zeros(len(in1), dtype=Wa.dtype).index_add_(0, gidx, (ub * ub).sum(0))
            res.append((in1 - out1, in2 - out2))
        return res

    def snapshot():
        return [(m.weight.detach().view(m.weight.shape[0], -1).double().clone(), m.bias.detach().double().clone())
                for m, _ in ch]

    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            w_prev = snapshot()
            if cfg["opt"] == "sam":
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                ps = [p for p in model.parameters()]
                gn = torch.norm(torch.stack([p.grad.norm() for p in ps])) + 1e-12
                es = []
                with torch.no_grad():
                    for p in ps:
                        e = p.grad * (rho / gn)
                        p.add_(e)
                        es.append(e)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                with torch.no_grad():
                    for p, e in zip(ps, es):
                        p.sub_(e)
                opt.step()
            else:
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            step += 1
            for l, (a, b) in enumerate(unit_terms(w_prev, snapshot())):
                cum1[l] += a
                cum2[l] += b
        q = [x.double() for x in R.conserved_quantity(model)]
        tr_loss, _ = M.loss_acc(model, xtr, ytr)
        log.append({
            "epoch": epoch, "step": step, "train_loss": tr_loss, "imbalance": R.imbalance(model),
            "first_order_mean": [float(c.mean()) for c in cum1],
            "second_order_mean": [float(c.mean()) for c in cum2],
            "first_order_absmean": [float(c.abs().mean()) for c in cum1],
            "dq_mean": [float((qq - q00).mean()) for qq, q00 in zip(q, q0)],
            "check_err": [float(((qq - q00) - (c1 + c2)).abs().max()) for qq, q00, c1, c2 in zip(q, q0, cum1, cum2)],
        })
    # effect on sharpness at the end
    X = xev
    dg = M.diag_ggn_linear(model, X)
    mb, sc = R.balance_min_norm(model)
    dgb = R.transport_diag(model, dg, sc)
    res = {"cfg": cfg, "log": log, "time": time.time() - t0,
           "ggn_trace": float(dg.sum()), "bal_ggn_trace": float(dgb.sum()),
           "orbitmin_trace": R.orbit_min_trace(model, dg)[0], "d_in": xtr[0].numel()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", action="store_true", help="other diagonal preconditioners (RMSprop, signSGD)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = []
    if a.extra:
        lrs = {"rmsprop": [3e-5, 1e-4, 3e-4], "signsgd": [3e-5, 1e-4, 3e-4]}
        grid = itertools.product([8, 32], [0, 1])
    else:
        lrs = {"sgd": [0.01, 0.03, 0.1], "sgdm": [0.001, 0.003, 0.01], "sam": [0.003],
               "adam": [3e-5, 1e-4, 3e-4, 1e-3]}
        grid = itertools.product([8, 16, 32], [0, 1])
    for res, seed in grid:
        for opt, lr_list in lrs.items():
            for lr in lr_list:
                cfgs.append(dict(opt=opt, lr=lr, bs=128, res=res, seed=seed, epochs=20))
    todo = [c for c in cfgs if not os.path.exists(os.path.join(OUT, f"{c['opt']}_lr{c['lr']}_res{c['res']}_s{c['seed']}.json"))]
    print(len(todo), "runs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                r = f.result()
            except Exception as e:  # keep going
                print("ERROR", c, repr(e), flush=True)
                continue
            with open(os.path.join(OUT, f"{c['opt']}_lr{c['lr']}_res{c['res']}_s{c['seed']}.json"), "w") as fh:
                json.dump(r, fh)
            last = r["log"][-1]
            print(f"[{i}/{len(todo)}] {c} loss={last['train_loss']:.3f} imb={[round(x, 3) for x in last['imbalance']]} "
                  f"dq1={[round(x, 4) for x in last['first_order_mean']]} dq2={[round(x, 4) for x in last['second_order_mean']]} "
                  f"err={max(last['check_err']):.1e} t={r['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
