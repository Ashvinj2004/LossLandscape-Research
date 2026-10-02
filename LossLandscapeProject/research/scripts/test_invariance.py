"""Which perturbation measures are invariant to which function-preserving rescalings?

Checked numerically on a briefly trained MLP (ReLU neuron rescaling) and a BatchNorm MLP (pre-BN row
scaling, and the ReLU rescaling between the BN affine parameters and the next layer). A measure is
reported invariant if it changes by less than 1e-3 (relative; BatchNorm's epsilon makes row
scaling invariance hold only up to ~1e-4, visible in the loss itself) under a random rescaling with log-scale
s.d. 1. Result: multiplicative average, ASAM and sum w^2 H_ii are invariant to all of them;
filter-normalized sharpness is invariant to row scaling only.

usage: python scripts/test_invariance.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls import data as D, measures as M  # noqa: E402
from lls.models import MLP, MLPBN  # noqa: E402


def measures(m, X, Y):
    L0 = M.loss_only(m, X, Y)
    r = {k: M.average_sharpness(m, X, Y, k, [0.03], n_draws=8, seed=0, L0=L0)[0.03][0] for k in ["iso", "filter", "mult"]}
    r["asam"] = M.worst_sharpness(m, X, Y, 0.5, adaptive=True, L0=L0)[0]
    r["wtrace"] = M.weighted_hessian_trace(m, X, Y, n_probes=8, seed=0)[0]
    r["loss"] = L0
    return r


def train(m, d, steps=300, lr=0.05):
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, len(d["ytr"]), (128,), generator=g)
        loss = F.cross_entropy(m(d["xtr"][idx]), d["ytr"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


def compare(name, m, X, Y, transform):
    a = measures(m, X, Y)
    with torch.no_grad():
        transform(m)
    b = measures(m, X, Y)
    print(f"\n{name}  (loss {a['loss']:.6f} -> {b['loss']:.6f})")
    for k in a:
        if k == "loss":
            continue
        rel = abs(b[k] - a[k]) / max(abs(a[k]), 1e-12)
        print(f"  {k:7s} {a[k]:.6g} -> {b[k]:.6g}   {'invariant' if rel < 1e-3 else f'CHANGES (x{b[k] / a[k]:.3g})'}")


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    d = D.load("mnist")
    X, Y = d["xev"][:1024], d["yev"][:1024]
    g = torch.Generator().manual_seed(1)

    m = MLP(784)
    train(m, d)
    lins = [l for l in m.modules() if isinstance(l, torch.nn.Linear)]

    def neuron(m):
        s = torch.exp(torch.randn(lins[0].out_features, generator=g))
        lins[0].weight.mul_(s[:, None]); lins[0].bias.mul_(s); lins[1].weight.div_(s[None, :])
    compare("MLP, ReLU neuron rescaling (layer 1)", m, X, Y, neuron)

    mb = MLPBN(784)
    train(mb, d)
    for bn in mb.modules():  # batch statistics, running stats frozen: BN scale invariance is exact
        if isinstance(bn, torch.nn.BatchNorm1d):
            bn.momentum = 0.0
    lb = [l for l in mb.modules() if isinstance(l, torch.nn.Linear)]
    bns = [b for b in mb.modules() if isinstance(b, torch.nn.BatchNorm1d)]

    def row(m):
        lb[0].weight.mul_(torch.exp(torch.randn(lb[0].out_features, generator=g))[:, None])
    compare("BN-MLP, pre-BN row scaling (layer 1)", mb, X, Y, row)

    def affine(m):
        s = torch.exp(torch.randn(bns[0].num_features, generator=g))
        bns[0].weight.mul_(s); bns[0].bias.mul_(s); lb[1].weight.div_(s[None, :])
    compare("BN-MLP, ReLU rescaling of BN affine vs next layer (layer 1)", mb, X, Y, affine)


if __name__ == "__main__":
    main()
