import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from lls import measures as M, rescale as R, data as D
from lls.models import SmallCNN
torch.manual_seed(0); torch.set_num_threads(2)
d = D.load("cifar10")
X = d["xev"][:128]
m = SmallCNN(width=8); opt = torch.optim.Adam(m.parameters(), 1e-3)
for i in range(0, 4000, 64):
    loss = F.cross_entropy(m(d["xtr"][i:i+64]), d["ytr"][i:i+64]); opt.zero_grad(); loss.backward(); opt.step()
print("chain:", [(type(mm).__name__, None if g is None else len(g)) for mm, g in R.chain(m)])
print("imbalance:", R.imbalance(m))
mb, sc = R.balance_min_norm(m)
print("balanced imbalance:", R.imbalance(mb))
with torch.no_grad(): print("max |f - f_bal|:", (m(X) - mb(X)).abs().max().item(), "scale", m(X).abs().max().item())
t = time.time(); dg = M.diag_ggn(m, X, chunk=64); print("diag_ggn time", time.time() - t)
dgb = M.diag_ggn(mb, X, chunk=64)
print("transport rel err:", ((R.transport_diag(m, dg, sc) - dgb).abs().sum() / dgb.abs().sum()).item())
for p in (1.0, 0.5):
    v, s2 = R.orbit_min_trace(m, dg, p=p)
    mm = R.apply_scales(m, s2); dgm = M.diag_ggn(mm, X, chunk=64)
    print(f"p={p}: raw {(dg.clamp_min(0)**p).sum():.4f} bal {(dgb.clamp_min(0)**p).sum():.4f} orbit-min {v:.4f} recomputed {(dgm.clamp_min(0)**p).sum():.4f}")
