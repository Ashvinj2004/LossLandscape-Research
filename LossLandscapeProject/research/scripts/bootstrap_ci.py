"""95% cluster-bootstrap CIs (resampling runs within each seed) for the headline Adam-vs-SGD statistics:
P(Adam flatter) for several measures, the function/coordinate split of log2(tr G ratio), and P_trG - P_MS1.

usage: python scripts/bootstrap_ci.py main_cifar10_mlp main_mnist_mlp
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls.analysis import load_sweep  # noqa: E402

COLS = {"tr": "ggn_trace", "ms": "orbitmin_trace", "orig": "orig_rel_sharp_test", "filt": "avg_filter_0.03"}


def P(x, y, sx, sy):
    w = n = 0
    for s in np.unique(sx):
        a, b = x[sx == s], y[sy == s]
        w += (a[:, None] < b[None, :]).sum()
        n += len(a) * len(b)
    return w / n


def stats(A, B):
    out = {f"P_{k}": P(A[k], B[k], A["seed"], B["seed"]) for k in COLS}
    out["function_bits"] = np.log2(A["ms"]).mean() - np.log2(B["ms"]).mean()
    out["coord_bits"] = np.log2(A["tr"] / A["ms"]).mean() - np.log2(B["tr"] / B["ms"]).mean()
    out["P_tr-P_ms"] = out["P_tr"] - out["P_ms"]
    return out


def main(names, other="sgd", B_=2000, seed=1):
    rng = np.random.default_rng(seed)

    def resample(D):
        idx = np.concatenate([rng.choice(np.where(D["seed"] == s)[0], (D["seed"] == s).sum())
                              for s in np.unique(D["seed"])])
        return {k: v[idx] for k, v in D.items()}

    for name in names:
        d = load_sweep(name)
        d = d[d.target.notna()]
        for t in [0.3, 0.1, 0.01]:
            A, B = [dict(seed=g.seed.values, **{k: g[c].values for k, c in COLS.items()})
                    for g in (d[(d.opt == "adam") & (d.target == t)], d[(d.opt == other) & (d.target == t)])]
            est = stats(A, B)
            bs = [stats(resample(A), resample(B)) for _ in range(B_)]
            print(f"{name} vs {other} @ loss {t}")
            for k, v in est.items():
                lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
                print(f"   {k:14s} {v:+.2f}  [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
