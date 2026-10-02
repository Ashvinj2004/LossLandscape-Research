"""Exactness check of the ResNet measures against a dense Hessian (tiny ResNet-18, width 1, float64).

The dense Hessian is assembled column by column from Hessian-vector products in batch-statistics mode,
then compared with: Lanczos lambda_max (30 steps), and the Hutchinson estimates of tr H, sum w^2 H_ii and
the BN-normalized trace (2000 probes; their errors should match the reported standard errors).

usage: python scripts/test_resnet_measures.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resnet_cifar10 as R  # noqa: E402


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    m = R.ResNet18(width=1).double()
    X = torch.randn(32, 3, 32, 32, dtype=torch.float64)
    Y = torch.randint(0, 10, (32,))
    with R.landscape_mode(m):
        n = R.get_flat(m).numel()
        cache = os.environ.get("HESS_CACHE")
        if cache and os.path.exists(cache):
            H = torch.load(cache)
        else:
            H = torch.zeros(n, n, dtype=torch.float64)
            for i in range(n):
                e = torch.zeros(n, dtype=torch.float64)
                e[i] = 1.0
                H[:, i] = R.hvp_flat(m, X, Y, e, 16)
            if cache:
                torch.save(H, cache)
        print(f"{n} parameters; Hessian symmetric to {float((H - H.T).abs().max() / H.abs().max()):.1e}")
        H = 0.5 * (H + H.T)
        ev = torch.linalg.eigvalsh(H)
        lam, conv = R.lambda_max(m, X, Y, 16, iters=30)
        print(f"lambda_max exact {float(ev[-1]):.6f}  Lanczos(30) {lam:.6f}  rel err {abs(lam - float(ev[-1])) / float(ev[-1]):.1e}")
        w = R.get_flat(m)
        d = R.conv_filter_weights(m).double()
        exact = {"hess_trace": float(H.diagonal().sum()), "wtrace": float((w * w * H.diagonal()).sum()),
                 "ntrace": float((d * H.diagonal()).sum())}
        est = R.quad_traces(m, X, Y, 16, 2000, 0, {"hess_trace": None, "wtrace": w * w, "ntrace": d})
        for k in exact:
            mu, se = est[k]
            print(f"{k:10s} exact {exact[k]:.6f}  Hutchinson(2000) {mu:.6f} +- {se:.6f}  "
                  f"({abs(mu - exact[k]) / se:.1f} s.e.)")


if __name__ == "__main__":
    main()
