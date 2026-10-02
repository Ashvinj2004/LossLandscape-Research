import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from lls import measures as M, rescale as R, data as D
from lls.models import MLP
torch.manual_seed(0); torch.set_num_threads(2)
d = D.load("cifar10")
X, Y = d["xev"][:1024], d["yev"][:1024]
m = MLP(3072); opt = torch.optim.Adam(m.parameters(), 1e-3)
for ep in range(3):
    perm = torch.randperm(10000)
    for i in range(0, 10000, 128):
        idx = perm[i:i+128]; loss = F.cross_entropy(m(d["xtr"][idx]), d["ytr"][idx]); opt.zero_grad(); loss.backward(); opt.step()
print("imbalance (mean |log in/out|):", R.imbalance(m))
mb, sc = R.balance_min_norm(m)
print("balanced imbalance:", R.imbalance(mb), "norm raw", M.get_flat(m).norm().item(), "balanced", M.get_flat(mb).norm().item())
with torch.no_grad(): print("max |f - f_bal|:", (m(X) - mb(X)).abs().max().item())
dg = M.diag_ggn_linear(m, X); dgb = M.diag_ggn_linear(mb, X)
dgt = R.transport_diag(m, dg, sc)
print("transport rel err:", ((dgt - dgb).abs().sum() / dgb.abs().sum()).item())
trmin, sc2 = R.orbit_min_trace(m, dg)
mm = R.apply_scales(m, sc2); dgm = M.diag_ggn_linear(mm, X)
with torch.no_grad(): print("max |f - f_min|:", (m(X) - mm(X)).abs().max().item())
print(f"trace raw {dg.sum():.3f} balanced {dgb.sum():.3f} orbit-min {trmin:.3f} recomputed-at-min {dgm.sum():.3f}")
# random rescalings must never beat orbit-min
for t in range(5):
    rs = [torch.exp(torch.randn(l.out_features) * 0.5) for l in R.linear_layers(m)[:-1]]
    print(" random-rescale trace", R.transport_diag(m, dg, rs).sum().item())
print("invariants: sum w^2 H_ii raw/bal", (dg * M.get_flat(m)**2).sum().item(), (dgb * M.get_flat(mb)**2).sum().item())
