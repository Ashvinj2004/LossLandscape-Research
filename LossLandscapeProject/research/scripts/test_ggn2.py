import sys, time, torch
sys.path.insert(0, ".")
from lls import measures as M
from lls.models import MLP
from lls import data as D
torch.manual_seed(0); torch.set_num_threads(2)
m = MLP(20, hidden=(8, 6), n_classes=10)
X = torch.randn(33, 20)
a, b = M.diag_ggn(m, X, chunk=7), M.diag_ggn_linear(m, X, chunk=10)
print("fast vs vmap max err", (a - b).abs().max().item(), a.abs().max().item())
d = D.load("cifar10"); m = MLP(3072)
t = time.time(); r = M.ggn_measures(m, d["xev"]); print("fast cifar time", time.time() - t, r)
