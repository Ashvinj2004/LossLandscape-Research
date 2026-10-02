"""'Same function, different landscape': 2-D isotropic loss slices around
(a) an Adam solution in its own coordinates, (b) the identical function after a function-preserving
rescaling (minimum-norm representative), (c) an SGD solution. Same random directions and scale.
Models are the final checkpoints of the main CIFAR-10 sweep (training loss <= 0.01)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lls import data as D, measures as M, rescale as R  # noqa: E402
from lls.models import build  # noqa: E402
from make_figures import save, INK2, MUTED  # noqa: E402

torch.set_num_threads(2)
RUNS = os.path.join(ROOT, "checkpoints")


def load(name):
    ck = torch.load(os.path.join(RUNS, name + ".pt"))
    m = build(ck["cfg"]["arch"], ck["cfg"]["dataset"])
    m.load_state_dict(ck["state"])
    return m.eval()


def surface(model, X, Y, d1, d2, alphas):
    w0 = M.get_flat(model)
    Z = np.zeros((len(alphas), len(alphas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            M.set_flat(model, w0 + a * d1 + b * d2)
            Z[j, i] = M.loss_only(model, X, Y)
    M.set_flat(model, w0)
    return Z


def main(adam="cifar10_mlp_adam_lr0.0006_bs128_s0", sgd="cifar10_mlp_sgd_lr0.06_bs128_s0", R_=30.0, n=31):
    d = D.load("cifar10")
    X, Y = d["xev"][:2048], d["yev"][:2048]
    ma, ms = load(adam), load(sgd)
    mb, _ = R.balance_min_norm(ma)
    with torch.no_grad():
        assert torch.allclose(ma(X), mb(X), atol=1e-4), "rescaling must preserve the function"
    g = torch.Generator().manual_seed(0)
    nparam = M.get_flat(ma).numel()
    d1 = torch.randn(nparam, generator=g)
    d2 = torch.randn(nparam, generator=g)
    d1, d2 = d1 / d1.norm(), d2 / d2.norm()
    alphas = np.linspace(-R_, R_, n)
    panels = [("(a) Adam, as trained", ma), ("(b) network (a), rescaled", mb),
              ("(c) SGD, as trained", ms)]
    Zs = [surface(m, X, Y, d1, d2, alphas) for _, m in panels]
    trs = [float(M.diag_ggn_linear(m, X).sum()) for _, m in panels]
    L0s = [Z[n // 2, n // 2] for Z in Zs]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    levels = np.linspace(0, 2.5, 11)
    for ax, (title, _), Z, tr, L0 in zip(axs, panels, Zs, trs, L0s):
        cs = ax.contourf(alphas, alphas, np.clip(Z - L0, 0, levels[-1]), levels=levels, cmap="Blues")
        ax.contour(alphas, alphas, Z - L0, levels=levels[1:], colors="white", linewidths=0.4)
        ax.plot(0, 0, marker="+", color=INK2, ms=6)
        ax.set_title(f"{title}\nHessian trace {tr:,.0f}", loc="left", fontsize=7)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("direction 1 (isotropic)")
    axs[0].set_ylabel("direction 2 (isotropic)")
    cb = fig.colorbar(cs, ax=axs, shrink=0.85, pad=0.02)
    cb.set_label("loss increase", color=INK2)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.5, -0.04, "(a) and (b) compute exactly the same function; their landscapes differ only through the "
             "coordinates.", ha="center", fontsize=7, color=MUTED)
    save(fig, "fig0_same_function")
    print("trace a/b/c:", [round(t) for t in trs], "L0:", [round(x, 4) for x in L0s])


if __name__ == "__main__":
    main()
