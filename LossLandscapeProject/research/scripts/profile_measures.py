import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from lls import measures as M, rescale as R, data as D
from lls.models import MLP
torch.manual_seed(0); torch.set_num_threads(2)
d = D.load("cifar10"); X, Y = d["xev"], d["yev"]
m = MLP(3072); opt = torch.optim.SGD(m.parameters(), 0.01, momentum=0.9)
for ep in range(5):
    perm = torch.randperm(10000)
    for i in range(0, 10000, 128):
        idx = perm[i:i+128]; loss = F.cross_entropy(m(d["xtr"][idx]), d["ytr"][idx]); opt.zero_grad(); loss.backward(); opt.step()
def T(name, f):
    t = time.time(); out = f(); print(f"{name:28s} {time.time()-t:6.2f}s"); return out
L0 = T("loss", lambda: M.loss_only(m, X, Y))
T("grad", lambda: M.grad_flat(m, X, Y))
v = torch.randn(M.get_flat(m).numel())
T("hvp", lambda: M.hvp_flat(m, X, Y, v))
T("lambda_max", lambda: M.hessian_top_eigs(m, X, Y))
T("trace 60", lambda: M.hessian_trace(m, X, Y, 60))
dg = T("diag_ggn", lambda: M.diag_ggn_linear(m, X))
T("orbit min p1", lambda: R.orbit_min_trace(m, dg, 1.0))
T("orbit min p.5", lambda: R.orbit_min_trace(m, dg, 0.5))
T("avg iso 3x16", lambda: M.average_sharpness(m, X, Y, "iso", [0.003, 0.01, 0.03], 16))
T("avg filter 3x16", lambda: M.average_sharpness(m, X, Y, "filter", [0.01, 0.03, 0.1], 16))
T("worst sam 20", lambda: M.worst_sharpness(m, X, Y, 0.05))
T("function_stats", lambda: M.function_stats(m, X, Y))
T("ALL full", lambda: M.all_measures(m, (X, Y), te=(d["xte"], d["yte"])))
