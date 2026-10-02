"""Loss-landscape geometry measures.

Conventions
-----------
* All measures are computed on a fixed evaluation subset (X, Y) of the training set,
  with mean cross-entropy loss L.
* "abs" = L(w + delta) - L(w);  "rel" = (L(w + delta) - L(w)) / L(w)  (the original study's metric).
* Perturbation families (Gaussian, averaged over draws, common random numbers across models):
    iso    : delta = sigma * eps                         (original study; not scale invariant)
    filter : per-output-unit rows rescaled to sigma*||w_row||, biases untouched (Li et al. 2018)
    mult   : delta = sigma * |w| * eps                   (elementwise-adaptive, ASAM-style average case)
* Worst-case sharpness: max_{||T^{-1} delta|| <= rho} L(w + delta) - L(w) by normalized PGD,
  with T = I ("sam") or T = |w| ("asam", Kwon et al. 2021).
* Hessian: top eigenvalues by Lanczos on Hessian-vector products; trace by Hutchinson.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, eigsh


def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]


def get_flat(model):
    return torch.cat([p.detach().reshape(-1) for p in params_of(model)]).clone()


def set_flat(model, v):
    i = 0
    with torch.no_grad():
        for p in params_of(model):
            n = p.numel()
            p.copy_(v[i:i + n].view_as(p))
            i += n


def _chunks(X, Y, chunk):
    for i in range(0, len(Y), chunk):
        yield X[i:i + chunk], Y[i:i + chunk]


@torch.no_grad()
def loss_acc(model, X, Y, chunk=4096):
    tot, correct = 0.0, 0
    for xb, yb in _chunks(X, Y, chunk):
        out = model(xb)
        tot += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return tot / len(Y), correct / len(Y)


def loss_only(model, X, Y, chunk=4096):
    return loss_acc(model, X, Y, chunk)[0]


def grad_flat(model, X, Y, chunk=4096):
    ps = params_of(model)
    g = [torch.zeros_like(p) for p in ps]
    N = len(Y)
    for xb, yb in _chunks(X, Y, chunk):
        loss = F.cross_entropy(model(xb), yb, reduction="sum") / N
        gs = torch.autograd.grad(loss, ps)
        for a, b in zip(g, gs):
            a.add_(b)
    return torch.cat([a.reshape(-1) for a in g])


def hvp_flat(model, X, Y, v, chunk=4096):
    ps = params_of(model)
    vs, i = [], 0
    for p in ps:
        vs.append(v[i:i + p.numel()].view_as(p))
        i += p.numel()
    out = [torch.zeros_like(p) for p in ps]
    N = len(Y)
    for xb, yb in _chunks(X, Y, chunk):
        loss = F.cross_entropy(model(xb), yb, reduction="sum") / N
        gs = torch.autograd.grad(loss, ps, create_graph=True)
        dot = sum((g * u).sum() for g, u in zip(gs, vs))
        hv = torch.autograd.grad(dot, ps)
        for a, b in zip(out, hv):
            a.add_(b)
    return torch.cat([a.reshape(-1) for a in out])


def hessian_top_eigs(model, X, Y, k=1, precond=None, tol=1e-3, chunk=4096):
    """Top-k eigenvalues of H (or of P^{-1/2} H P^{-1/2} if precond=P given as flat vector)."""
    n = get_flat(model).numel()
    s = None if precond is None else precond.rsqrt()

    def mv(x):
        v = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1))
        if s is not None:
            v = v * s
        hv = hvp_flat(model, X, Y, v, chunk)
        if s is not None:
            hv = hv * s
        return hv.double().numpy()

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    vals = eigsh(op, k=k, which="LA", tol=tol, ncv=max(20, 2 * k + 1), return_eigenvectors=False)
    return sorted(vals.tolist(), reverse=True)


def hessian_trace(model, X, Y, n_probes=50, seed=0, chunk=4096):
    g = torch.Generator().manual_seed(seed)
    n = get_flat(model).numel()
    vals = []
    for _ in range(n_probes):
        z = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
        vals.append(torch.dot(z, hvp_flat(model, X, Y, z, chunk)).item())
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _layer_slices(model):
    out, i = [], 0
    for p in params_of(model):
        out.append((p.shape, i, i + p.numel()))
        i += p.numel()
    return out


def perturbation_direction(model, w0, kind, gen):
    eps = torch.randn(w0.numel(), generator=gen)
    if kind == "iso":
        return eps
    if kind == "mult":
        return eps * w0.abs()
    if kind == "filter":
        d = torch.zeros_like(w0)
        for shape, a, b in _layer_slices(model):
            if len(shape) == 1:  # biases untouched (Li et al. 2018, ignore='biasbn')
                continue
            e = eps[a:b].view(shape[0], -1)
            w = w0[a:b].view(shape[0], -1)
            d[a:b] = (e / (e.norm(dim=1, keepdim=True) + 1e-12) * w.norm(dim=1, keepdim=True)).reshape(-1)
        return d
    raise ValueError(kind)


def average_sharpness(model, X, Y, kind, sigmas, n_draws=16, seed=0, L0=None):
    """Expected loss increase under random perturbations, for each sigma."""
    w0 = get_flat(model)
    L0 = loss_only(model, X, Y) if L0 is None else L0
    out = {}
    for s in sigmas:
        g = torch.Generator().manual_seed(seed)  # common random numbers across sigmas & models
        incs = []
        for _ in range(n_draws):
            d = perturbation_direction(model, w0, kind, g)
            set_flat(model, w0 + s * d)
            incs.append(loss_only(model, X, Y) - L0)
        set_flat(model, w0)
        incs = np.array(incs)
        out[s] = (float(incs.mean()), float(incs.std(ddof=1) / math.sqrt(n_draws)))
    return out


def worst_sharpness(model, X, Y, rho, adaptive=False, steps=10, L0=None):
    """max_{||u|| <= rho} L(w0 + T u) - L(w0), T = |w0| if adaptive else I. Returns (pgd, one_step)."""
    w0 = get_flat(model)
    L0 = loss_only(model, X, Y) if L0 is None else L0
    T = w0.abs() if adaptive else torch.ones_like(w0)
    u = torch.zeros_like(w0)
    alpha = 4.0 * rho / steps  # reaches the boundary in ~steps/4 steps, then moves along it
    one_step = None
    best = 0.0
    for t in range(steps):
        set_flat(model, w0 + T * u)
        g = grad_flat(model, X, Y) * T
        gn = g.norm()
        if gn < 1e-20:
            break
        if t == 0:
            set_flat(model, w0 + T * (rho * g / gn))
            one_step = loss_only(model, X, Y) - L0
        u = u + alpha * g / gn
        un = u.norm()
        if un > rho:
            u = u * (rho / un)
        set_flat(model, w0 + T * u)
        best = max(best, loss_only(model, X, Y) - L0)
    set_flat(model, w0)
    return float(best), float(one_step if one_step is not None else 0.0)


def weight_stats(model, w_init=None):
    w0 = get_flat(model)
    d = {"w_norm": float(w0.norm())}
    layer_norms = []
    for p in params_of(model):
        if p.dim() > 1:
            layer_norms.append(float(p.detach().norm()))
    d["layer_norms"] = layer_norms
    d["log_prod_layer_norms"] = float(np.sum(np.log(layer_norms)))
    if w_init is not None:
        d["dist_init"] = float((w0 - w_init).norm())
    return d


def function_stats(model, X, Y, chunk=4096):
    """Margins, confidence, input-gradient norm, gradient norm."""
    margins, conf, ig = [], [], []
    for xb, yb in _chunks(X, Y, chunk):
        xb = xb.clone().requires_grad_(True)
        out = model(xb)
        loss = F.cross_entropy(out, yb, reduction="sum")
        gx, = torch.autograd.grad(loss, xb)
        ig.append(gx.flatten(1).norm(dim=1).detach())
        out = out.detach()
        true = out.gather(1, yb[:, None]).squeeze(1)
        other = out.clone()
        other.scatter_(1, yb[:, None], -float("inf"))
        margins.append(true - other.max(1).values)
        conf.append(F.softmax(out, 1).gather(1, yb[:, None]).squeeze(1))
    m = torch.cat(margins)
    return {
        "margin_median": float(m.median()),
        "margin_q10": float(m.quantile(0.1)),
        "p_true_mean": float(torch.cat(conf).mean()),
        "input_grad_norm": float(torch.cat(ig).mean()),
        "grad_norm": float(grad_flat(model, X, Y).norm()),
    }


def diag_ggn(model, X, chunk=128):
    """Exact diagonal of the Gauss-Newton matrix for softmax cross-entropy (= true Fisher):
        diag G = (1/N) sum_n sum_k p_nk (d log p_nk / d theta)^2.
    Label-free; equals diag(H) up to the residual term, which vanishes as the loss -> 0."""
    from torch.func import functional_call, grad, vmap
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    acc = {n: torch.zeros_like(p) for n, p in params.items()}
    N = len(X)
    for i in range(0, N, chunk):
        xb = X[i:i + chunk]
        with torch.no_grad():
            probs = F.softmax(model(xb), 1)
        K = probs.shape[1]
        for k in range(K):
            def logp(p, x, k=k):
                return F.log_softmax(functional_call(model, p, (x.unsqueeze(0),)), -1)[0, k]
            gs = vmap(grad(logp), in_dims=(None, 0))(params, xb)
            w = probs[:, k]
            for n in names:
                g2 = gs[n] ** 2
                acc[n] += (w.view(-1, *[1] * (g2.dim() - 1)) * g2).sum(0)
    return torch.cat([acc[n].reshape(-1) / N for n in names])


def diag_ggn_linear(model, X, chunk=4096):
    """Same quantity as diag_ggn, fast path for networks whose parameters all live in nn.Linear:
    per-sample grad of W is delta a^T, so sum_n p_nk (delta_n^2)(a_n^2)^T is one matmul per class."""
    lins = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert sum(p.numel() for m in lins for p in m.parameters()) == sum(p.numel() for p in params_of(model))
    accW = [torch.zeros_like(m.weight) for m in lins]
    accb = [torch.zeros_like(m.bias) for m in lins]
    store = {}

    def fwd_hook(mod, inp, out):
        store[mod] = (inp[0].detach(), out)
        out.retain_grad()

    hs = [m.register_forward_hook(fwd_hook) for m in lins]
    N = len(X)
    try:
        for i in range(0, N, chunk):
            xb = X[i:i + chunk]
            for k in range(10):
                store.clear()
                logp = F.log_softmax(model(xb), 1)
                p = logp.detach().exp()[:, k]
                logp[:, k].sum().backward()
                for j, m in enumerate(lins):
                    a, z = store[m]
                    d2 = z.grad ** 2 * p[:, None]
                    accW[j] += d2.T @ (a ** 2)
                    accb[j] += d2.sum(0)
                model.zero_grad(set_to_none=True)
    finally:
        for h in hs:
            h.remove()
    parts = []
    for m, W, b in zip(lins, accW, accb):  # match params_of(model) ordering (weight, bias per layer)
        parts += [W.reshape(-1) / N, b.reshape(-1) / N]
    return torch.cat(parts)


def is_mlp(model):
    return all(isinstance(m, (torch.nn.Linear, torch.nn.ReLU, torch.nn.Sequential, torch.nn.Flatten))
               or m is model for m in model.modules())


def ggn_measures(model, X, w0=None, dg=None):
    """Optimizer-specific sharpness: SGD's tr(H) vs Adam's tr(diag(H)^{1/2}) (Li, Wen & Lyu 2025)."""
    if dg is None:
        dg = diag_ggn_linear(model, X) if is_mlp(model) else diag_ggn(model, X)
    w0 = get_flat(model) if w0 is None else w0
    return {
        "ggn_trace": float(dg.sum()),
        "ggn_sqrt_trace": float(dg.clamp_min(0).sqrt().sum()),
        # rescaling-invariant variants: curvature in units of the weights (diag(|w|) H diag(|w|))
        "ggn_trace_wscaled": float((dg * w0 ** 2).sum()),
        "ggn_sqrt_trace_wscaled": float((dg.clamp_min(0).sqrt() * w0.abs()).sum()),
    }


def rescaling_measures(model, X, Y, dg, seed=0, level="full"):
    """Measures at canonical points of the ReLU rescaling orbit (MLPs only)."""
    from . import rescale as R
    r = {}
    for l, v in enumerate(R.imbalance(model)):
        r[f"imbalance_{l}"] = v
    mb, sc = R.balance_min_norm(model)
    dgb = R.transport_diag(model, dg, sc)
    r["bal_w_norm"] = float(get_flat(mb).norm())
    r["bal_ggn_trace"] = float(dgb.sum())
    r["bal_ggn_sqrt_trace"] = float(dgb.clamp_min(0).sqrt().sum())
    r["orbitmin_trace"], _ = R.orbit_min_trace(model, dg, p=1.0)
    r["orbitmin_sqrt_trace"], _ = R.orbit_min_trace(model, dg, p=0.5)
    if level != "lean":
        r["bal_lambda_max"] = hessian_top_eigs(mb, X, Y, k=1)[0]
    L0 = loss_only(mb, X, Y)
    res = average_sharpness(mb, X, Y, "iso", SIGMAS["iso"], n_draws=16 if level == "full" else 8, seed=seed, L0=L0)
    for s, (m, se) in res.items():
        r[f"bal_avg_iso_{s}"] = m
        r[f"bal_avg_iso_{s}_se"] = se
    if level == "full":
        for rho in RHOS["sam"]:
            r[f"bal_worst_sam_{rho}"], _ = worst_sharpness(mb, X, Y, rho, adaptive=False, L0=L0)
    return r


SIGMAS = {
    "iso": [0.003, 0.01, 0.03],
    "filter": [0.01, 0.03, 0.1],
    "mult": [0.01, 0.03, 0.1],
}
RHOS = {"sam": [0.05], "asam": [0.5]}


def has_batchnorm(model):
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


class landscape_mode:
    """For BatchNorm networks, measure the loss the optimizer actually sees: batch statistics (train
    mode), with running averages frozen (momentum 0) so measuring does not change the model. In this
    mode pre-BN weight rows are exactly scale-invariant. No-op for networks without BatchNorm."""

    def __init__(self, model):
        self.model = model
        self.bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        if self.bns:
            self.was = self.model.training
            self.moms = [m.momentum for m in self.bns]
            self.model.train()
            for m in self.bns:
                m.momentum = 0.0

    def __exit__(self, *exc):
        if self.bns:
            for m, mo in zip(self.bns, self.moms):
                m.momentum = mo
            self.model.train(self.was)


def weighted_hessian_trace(model, X, Y, n_probes=40, seed=0):
    """Hutchinson estimate of tr(diag|w| H diag|w|) = sum_i w_i^2 H_ii, which is invariant to every
    per-weight rescaling (both BatchNorm scale invariance and ReLU neuron rescaling)."""
    g = torch.Generator().manual_seed(seed)
    w = get_flat(model).abs()
    vals = []
    for _ in range(n_probes):
        v = (torch.randint(0, 2, (w.numel(),), generator=g).float() * 2 - 1) * w
        vals.append(torch.dot(v, hvp_flat(model, X, Y, v)).item())
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))


def bn_measures(model, X, Y, te=None, w_init=None, seed=0):
    """Measures for BatchNorm networks (no diag-GGN / orbit machinery: BN couples samples and adds a
    second symmetry family). Raw: trace, lambda_max, isotropic, SAM. Invariant to all per-weight
    rescalings: weighted trace, multiplicative and filter-normalized averages, ASAM."""
    L0, A0 = loss_acc(model, X, Y)
    r = {"ev_loss": L0, "ev_acc": A0}
    r.update(weight_stats(model, w_init))
    r.update(function_stats(model, X, Y))
    pre_bn = [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear) and m.bias is None]
    for l, W in enumerate(pre_bn):
        r[f"prebn_row_norm_{l}"] = float(W.detach().norm(dim=1).median())
    r["lambda_max"] = hessian_top_eigs(model, X, Y, k=1)[0]
    r["hess_trace"], r["hess_trace_se"] = hessian_trace(model, X, Y, n_probes=40, seed=seed)
    r["wtrace"], r["wtrace_se"] = weighted_hessian_trace(model, X, Y, n_probes=40, seed=seed)
    for kind, sigmas in SIGMAS.items():
        res = average_sharpness(model, X, Y, kind, sigmas, n_draws=16, seed=seed, L0=L0)
        for s, (m, se) in res.items():
            r[f"avg_{kind}_{s}"] = m
            r[f"avg_{kind}_{s}_se"] = se
    for name, rhos in RHOS.items():
        for rho in rhos:
            pgd, one = worst_sharpness(model, X, Y, rho, adaptive=(name == "asam"), L0=L0)
            r[f"worst_{name}_{rho}"] = pgd
    if te is not None:
        Lt = loss_only(model, *te)
        res = average_sharpness(model, te[0], te[1], "iso", [0.01], n_draws=10, seed=seed + 7, L0=Lt)
        r["orig_rel_sharp_test"] = res[0.01][0] / Lt
    return r


def all_measures(model, ev, te=None, w_init=None, precond=None, level="full", seed=0):
    """Compute the battery of measures. ev = (X, Y) eval subset of train; te = test set (optional)."""
    if has_batchnorm(model):
        with landscape_mode(model):
            return bn_measures(model, ev[0], ev[1], te=te, w_init=w_init, seed=seed)
    X, Y = ev
    L0, A0 = loss_acc(model, X, Y)
    r = {"ev_loss": L0, "ev_acc": A0}
    r.update(weight_stats(model, w_init))
    r.update(function_stats(model, X, Y))
    eigs = hessian_top_eigs(model, X, Y, k=1, tol=1e-3 if level != "lean" else 1e-2)
    r["lambda_max"] = eigs[0]
    if precond is not None and level != "lean":
        r["lambda_max_precond"] = hessian_top_eigs(model, X, Y, k=1, precond=precond)[0]
    if level != "lean":
        tr, tr_se = hessian_trace(model, X, Y, n_probes=20, seed=seed)  # cross-check of the exact GGN trace
        r["hess_trace"], r["hess_trace_se"] = tr, tr_se
    # exact diag-GGN: fast closed form for MLPs; per-sample vmap (on a 512-sample subset) otherwise
    dg = diag_ggn_linear(model, X) if is_mlp(model) else diag_ggn(model, X[:512], chunk=64)
    r.update(ggn_measures(model, X, dg=dg))
    r.update(rescaling_measures(model, X, Y, dg, seed=seed, level=level))
    for kind, sigmas in SIGMAS.items():
        res = average_sharpness(model, X, Y, kind, sigmas, n_draws=16 if level == "full" else 8, seed=seed, L0=L0)
        for s, (m, se) in res.items():
            r[f"avg_{kind}_{s}"] = m
            r[f"avg_{kind}_{s}_se"] = se
    if level == "full":
        for name, rhos in RHOS.items():
            for rho in rhos:
                pgd, one = worst_sharpness(model, X, Y, rho, adaptive=(name == "asam"), L0=L0)
                r[f"worst_{name}_{rho}"] = pgd
                r[f"onestep_{name}_{rho}"] = one
    if te is not None:
        # The original study's metric: isotropic sigma=0.01, relative, on the TEST set, 10 draws.
        Lt = loss_only(model, *te)
        res = average_sharpness(model, te[0], te[1], "iso", [0.01], n_draws=10, seed=seed + 7, L0=Lt)
        r["orig_rel_sharp_test"] = res[0.01][0] / Lt
    return r
