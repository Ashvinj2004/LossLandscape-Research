import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from lls import measures as M
from lls.models import MLP
torch.manual_seed(0); torch.set_num_threads(2)
# tiny brute-force check
m = MLP(5, hidden=(4, 3), n_classes=3)
X = torch.randn(7, 5)
dg = M.diag_ggn(m, X, chunk=3)
ps = list(m.parameters())
G = 0
for n in range(len(X)):
    out = m(X[n:n+1])[0]
    p = F.softmax(out, 0)
    J = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(out[k], ps, retain_graph=True)]) for k in range(3)])
    S = torch.diag(p) - torch.outer(p, p)
    G = G + J.T @ S @ J
G = G.detach() / len(X)
print("max abs err vs brute force:", (torch.diag(G) - dg).abs().max().item(), "scale", torch.diag(G).abs().max().item())
# timing at scale
from lls import data as D
d = D.load("cifar10")
m = MLP(3072)
t = time.time(); r = M.ggn_measures(m, d["xev"]); print("cifar mlp ggn time", time.time() - t, r)
