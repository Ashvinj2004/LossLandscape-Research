"""Is diag(H) = diag(G) for the networks in the paper?

For a ReLU network without normalization layers, the output is (almost everywhere) affine in each single
weight when all other weights are held fixed, so d^2 f / d w_i^2 = 0 and the residual term of the Hessian,
sum_k dL/df_k d^2 f_k / d w_i^2, has a zero diagonal. Hence diag(H) = diag(G) and tr H = tr G exactly,
even though H != G off the diagonal. This script checks it in float64 against the exact Hessian diagonal
(one Hessian-vector product per parameter) for an MLP and the small CNN, and shows that it fails for the
BatchNorm MLP, where normalization makes the output nonlinear in a single weight.

usage: python scripts/test_hessian_diag.py
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from lls import measures as M
from lls.models import MLP, MLPBN, SmallCNN


def exact_diag_hessian(m, X, Y):
    ps = [p for p in m.parameters() if p.requires_grad]
    loss = F.cross_entropy(m(X), Y)
    g = torch.cat([t.reshape(-1) for t in torch.autograd.grad(loss, ps, create_graph=True)])
    d = torch.zeros_like(g)
    for i in range(g.numel()):
        row = torch.autograd.grad(g[i], ps, retain_graph=True)
        d[i] = torch.cat([t.reshape(-1) for t in row])[i]
    return d.detach()


def brute_diag_ggn(m, X):
    """diag(J^T S J) from a full-batch forward pass (so that it is also right with batch statistics)."""
    ps = [p for p in m.parameters() if p.requires_grad]
    out = m(X)
    G = 0
    for n in range(len(X)):
        p = F.softmax(out[n], 0).detach()
        J = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(out[n, k], ps, retain_graph=True)])
                         for k in range(out.shape[1])])
        G = G + J.T @ (torch.diag(p) - torch.outer(p, p)) @ J
    return torch.diag(G).detach() / len(X)


def check(name, m, X, Y):
    m = m.double()
    X = X.double()
    dh = exact_diag_hessian(m, X, Y)
    dg = brute_diag_ggn(m, X)
    rel = (dh - dg).abs().max().item() / dh.abs().max().item()
    print(f"{name:10s} {dh.numel():5d} params  tr H {dh.sum().item():.6f}  tr G {dg.sum().item():.6f}  "
          f"max |diag H - diag G| / max |diag H| = {rel:.1e}")
    return rel


def main():
    torch.manual_seed(0)
    X = torch.randn(64, 20)
    Y = torch.randint(0, 10, (64,))
    ok = check("MLP", MLP(20, hidden=(16, 12), n_classes=10), X, Y) < 1e-10
    Xc = torch.randn(16, 3, 8, 8)
    Yc = torch.randint(0, 10, (16,))
    ok &= check("CNN", SmallCNN(width=4, img=8), Xc, Yc) < 1e-10
    bn = MLPBN(20, hidden=(16, 12), n_classes=10)
    bn.train()                      # batch statistics, as in all BN measurements
    rel_bn = check("BN-MLP", bn, X, Y)
    print("MLP, CNN: diag(H) = diag(G) exactly" if ok else "MLP/CNN: MISMATCH",
          "| BN-MLP: diag(H) != diag(G)" if rel_bn > 1e-6 else "| BN-MLP: equal")


if __name__ == "__main__":
    main()
