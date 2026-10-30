"""How exact is the BatchNorm scale invariance in ResNet-18 with BatchNorm's default epsilon (1e-5)?

Every conv filter is rescaled by a random factor (log-uniform in [0.3, 3], spanning the range of filter
norms seen in the trained runs relative to initialization), and loss, raw tr H, the BN-invariant trace
at unit-norm filters (ntrace) and sum w^2 H_ii are recomputed in float64 with the same Hutchinson probes
(their estimators are then exactly invariant if the symmetry is exact). Run at initialization and after
a short training phase, on 64 CIFAR-10 training images with batch statistics, as in the measurements.

usage: python scripts/test_resnet_bn_eps.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resnet_cifar10 as R  # noqa: E402


def measures(m, X, Y):
    with R.landscape_mode(m):
        L = float(F.cross_entropy(m(X), Y))
        w = R.get_flat(m)
        est = R.quad_traces(m, X, Y, 64, 4, 0, {"trH": None, "ntrace": R.conv_filter_weights(m).to(w.dtype), "wtrace": w * w})
    return {"loss": L, **{k: v[0] for k, v in est.items()}}


def rescale(m, gen):
    with torch.no_grad():
        for c in m.modules():
            if isinstance(c, nn.Conv2d):
                s = torch.exp(torch.empty(c.weight.shape[0], dtype=c.weight.dtype).uniform_(np.log(0.3), np.log(3), generator=gen))
                c.weight.mul_(s.view(-1, 1, 1, 1))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    xtr, ytr = R.load_cifar10(root)[:2]
    X = torch.tensor(xtr[:64]).double().div(255)
    X = (X - X.mean((0, 2, 3), keepdim=True)) / X.std((0, 2, 3), keepdim=True)
    Y = torch.tensor(ytr[:64])
    m = R.ResNet18().double()
    for phase in ["initialization", "after 30 Adam steps"]:
        if phase != "initialization":
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            for _ in range(30):
                opt.zero_grad()
                F.cross_entropy(m(X), Y).backward()
                opt.step()
        a = measures(m, X, Y)
        m2 = R.ResNet18().double()
        m2.load_state_dict(m.state_dict())
        rescale(m2, torch.Generator().manual_seed(1))
        b = measures(m2, X, Y)
        print(f"{phase}: " + "  ".join(f"{k} rel. change {abs(b[k] - a[k]) / abs(a[k]):.1e}" for k in a), flush=True)


if __name__ == "__main__":
    main()
