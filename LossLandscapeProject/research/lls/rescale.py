"""Function-preserving neuron/channel rescalings for ReLU networks (MLPs and conv nets).

For hidden unit j of layer l (a neuron, or a conv channel), scaling its incoming weights and bias
by c > 0 and its outgoing weights by 1/c leaves the network function unchanged (positive
homogeneity of ReLU and max-pooling). The set of all such rescalings is the network's
*rescaling orbit*. Euclidean sharpness measures are not constant on the orbit (Dinh et al., 2017),
so two networks can only be compared fairly after mapping both to a canonical point of their orbits.

Canonical points implemented here:
  * balance_min_norm : the minimum-L2-norm point (incoming norm == outgoing norm per unit);
  * orbit_min_trace  : the point minimizing sum_i diag(H)_ii^p, by convex coordinate descent.
Under a linear reparameterization w' = D w the Hessian transforms exactly as D^{-1} H D^{-1}, so
diag(H) is transported analytically instead of being recomputed.

Every layer is viewed as a 2-D matrix W (out x in_flat) plus bias, with `group[i]` giving the
previous layer's unit that input column i belongs to (identity for Linear->Linear, kernel
positions for conv, spatial positions after Flatten).
"""
import copy

import torch
import torch.nn as nn


def chain(model):
    """[(module, group_index or None)] for the rescalable chain of the model."""
    mods = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    out = []
    prev = None
    for m in mods:
        if prev is None:
            g = None
        else:
            C = prev.out_channels if isinstance(prev, nn.Conv2d) else prev.out_features
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1]
                g = torch.arange(C).repeat_interleave(k)
            else:
                n_in = m.in_features
                g = torch.arange(C).repeat_interleave(n_in // C)  # Flatten: channel-major
        out.append((m, g))
        prev = m
    return out


def linear_layers(model):
    return [m for m, _ in chain(model)]


def _W2(m):
    return m.weight.view(m.weight.shape[0], -1)


def _n_units(m):
    return m.out_channels if isinstance(m, nn.Conv2d) else m.out_features


@torch.no_grad()
def _scale_unit(ch, l, c):
    """In place: units of layer l scaled by c (incoming), compensated in layer l+1 (outgoing)."""
    m, _ = ch[l]
    nxt, g = ch[l + 1]
    _W2(m).mul_(c[:, None])
    m.bias.mul_(c)
    _W2(nxt).div_(c[g][None, :])


@torch.no_grad()
def apply_scales(model, scales):
    m2 = copy.deepcopy(model)
    ch = chain(m2)
    for l, c in enumerate(scales):
        _scale_unit(ch, l, c)
    return m2


@torch.no_grad()
def _in_out_norm2(ch, l):
    m, _ = ch[l]
    nxt, g = ch[l + 1]
    n_in = _W2(m).pow(2).sum(1) + m.bias.pow(2)
    n_out = torch.zeros(_n_units(m)).index_add_(0, g, _W2(nxt).pow(2).sum(0))
    return n_in, n_out


@torch.no_grad()
def balance_min_norm(model, iters=300, tol=1e-7):
    """Map to the minimum-norm point of the rescaling orbit. Returns (new_model, scales)."""
    m2 = copy.deepcopy(model)
    ch = chain(m2)
    total = [torch.ones(_n_units(m)) for m, _ in ch[:-1]]
    for _ in range(iters):
        worst = 0.0
        for l in range(len(ch) - 1):
            n_in, n_out = _in_out_norm2(ch, l)
            c = (n_out.clamp_min(1e-30) / n_in.clamp_min(1e-30)).pow(0.25)
            _scale_unit(ch, l, c)
            total[l] *= c
            worst = max(worst, float(c.log().abs().max()))
        if worst < tol:
            break
    return m2, total


def _split_diag(model, dg):
    """Split a flat diag vector (parameter order) into per-layer 2-D (W_diag, b_diag)."""
    out, i = [], 0
    for m, _ in chain(model):
        nW, nb = m.weight.numel(), m.bias.numel()
        out.append((dg[i:i + nW].view(m.weight.shape[0], -1), dg[i + nW:i + nW + nb]))
        i += nW + nb
    return out


def _scale_factors(ch, l, scales, dtype):
    """Per-entry multiplicative factor of layer l's weights under `scales` (out x in_flat)."""
    m, g = ch[l]
    W = _W2(m)
    L = len(ch)
    c_out = scales[l].to(dtype) if l < L - 1 else torch.ones(W.shape[0], dtype=dtype)
    c_in = scales[l - 1].to(dtype)[g] if l > 0 else torch.ones(W.shape[1], dtype=dtype)
    return c_out[:, None] / c_in[None, :], c_out


@torch.no_grad()
def transport_diag(model, dg, scales):
    """diag H at the rescaled point: an entry for w -> w * s scales as 1/s^2."""
    ch = chain(model)
    new = []
    for l, (DW, Db) in enumerate(_split_diag(model, dg)):
        s, c_out = _scale_factors(ch, l, scales, DW.dtype)
        new += [(DW / s ** 2).reshape(-1), Db / c_out ** 2]
    return torch.cat(new)


@torch.no_grad()
def orbit_min_trace(model, dg, p=1.0, iters=1000, tol=1e-10):
    """min over the rescaling orbit of sum_i diag(H)_ii^p  (p=1: tr H; p=1/2: Adam's tr diag(H)^{1/2}).
    A weight scaled by s has curvature^p scaled by s^{-2p}: the objective is a posynomial in the
    scales, hence convex in log-scales. Exact coordinate descent: unit j collects A_j c_j^{-2p}
    (incoming + bias) + B_j c_j^{2p} (outgoing), minimized at c_j^{2p} = sqrt(A_j / B_j)."""
    ch = chain(model)
    L = len(ch)
    D = [(W.double().clamp_min(0) ** p, b.double().clamp_min(0) ** p) for W, b in _split_diag(model, dg)]
    scales = [torch.ones(_n_units(m), dtype=torch.float64) for m, _ in ch[:-1]]

    def objective():
        tot = 0.0
        for l, (DW, Db) in enumerate(D):
            s, c_out = _scale_factors(ch, l, scales, torch.float64)
            tot += float((DW / s ** (2 * p)).sum() + (Db / c_out ** (2 * p)).sum())
        return tot

    prev = None
    for _ in range(iters):
        for l in range(L - 1):
            DW, Db = D[l]
            g_l = ch[l][1]
            c_in = scales[l - 1][g_l] if l > 0 else torch.ones(DW.shape[1], dtype=torch.float64)
            A = (DW * c_in[None, :] ** (2 * p)).sum(1) + Db
            DW2, _ = D[l + 1]
            g = ch[l + 1][1]
            c_next = scales[l + 1] if l + 1 < L - 1 else torch.ones(DW2.shape[0], dtype=torch.float64)
            B = torch.zeros(len(scales[l]), dtype=torch.float64).index_add_(
                0, g, (DW2 / c_next[:, None] ** (2 * p)).sum(0))
            scales[l] = (A.clamp_min(1e-300) / B.clamp_min(1e-300)).pow(1.0 / (4 * p))
        val = objective()
        if prev is not None and abs(prev - val) <= tol * abs(val):
            break
        prev = val
    return val, [s.float() for s in scales]


@torch.no_grad()
def _transform_state(opt, p, s):
    """Covariant transform of optimizer state for a parameter rescaled elementwise by s (w -> w*s):
    gradients scale 1/s, so first moments / momentum buffers scale 1/s and second moments 1/s^2."""
    if opt is None:
        return
    st = opt.state.get(p)
    if not st:
        return
    s = s.expand_as(p) if s.shape != p.shape else s
    if "exp_avg" in st:
        st["exp_avg"].div_(s)
    if "exp_avg_sq" in st:
        st["exp_avg_sq"].div_(s ** 2)
    if "square_avg" in st:  # RMSprop
        st["square_avg"].div_(s ** 2)
    if st.get("momentum_buffer") is not None:
        st["momentum_buffer"].div_(s)


@torch.no_grad()
def scale_units_inplace(model, scales, opt=None, layers=None):
    """Apply a function-preserving rescaling in place (layer l's units scaled by scales[l]),
    transforming the optimizer state covariantly."""
    ch = chain(model)
    for l, c in enumerate(scales):
        if layers is not None and l not in layers:
            continue
        m, _ = ch[l]
        nxt, g = ch[l + 1]
        sw = c.view(-1, *[1] * (m.weight.dim() - 1))
        _scale_unit(ch, l, c)
        _transform_state(opt, m.weight, sw.expand_as(m.weight))
        _transform_state(opt, m.bias, c)
        so = (1.0 / c[g]).view(1, -1).expand(nxt.weight.shape[0], -1).reshape(nxt.weight.shape)
        _transform_state(opt, nxt.weight, so)


@torch.no_grad()
def curvature_balance(model, dg, p=1.0):
    """Per hidden layer: mean and mean-|.| of log(A_j / B_j), where A_j = sum of incoming diag(G)^p and
    B_j = sum of outgoing diag(G)^p at the CURRENT coordinates. A_j = B_j for every unit is the
    stationarity condition of the orbit minimum of sum_i diag(G)_ii^p (p=1: minimum sharpness)."""
    ch = chain(model)
    parts = _split_diag(model, dg)
    out = []
    for l in range(len(ch) - 1):
        DW, Db = parts[l]
        A = (DW.clamp_min(0) ** p).sum(1) + Db.clamp_min(0) ** p
        DW2, _ = parts[l + 1]
        g = ch[l + 1][1]
        B = torch.zeros_like(A).index_add_(0, g, (DW2.clamp_min(0) ** p).sum(0))
        r = (A.clamp_min(1e-30) / B.clamp_min(1e-30)).log()
        out.append((float(r.mean()), float(r.abs().mean())))
    return out


@torch.no_grad()
def restore_conserved(model, q0, opt=None, sweeps=2, layers=None):
    """Function-preserving rescaling that resets each unit's Q_j = ||in||^2 - ||out||^2 to q0[l][j]
    (undoing any drift along the rescaling orbit). Solves c^2 n_in - n_out / c^2 = q0 for c^2 > 0.
    If an Adam optimizer is given, its moment estimates are transformed covariantly
    (grad of w -> w*s scales 1/s, so m -> m/s and v -> v/s^2)."""
    ch = chain(model)
    for _ in range(sweeps):
        for l in range(len(ch) - 1):
            if layers is not None and l not in layers:
                continue
            n_in, n_out = _in_out_norm2(ch, l)
            q = q0[l]
            x = (q + torch.sqrt(q * q + 4 * n_in * n_out)) / (2 * n_in.clamp_min(1e-30))
            c = x.clamp_min(1e-30).sqrt()
            _scale_unit(ch, l, c)
            if opt is not None:
                m, nxt = ch[l][0], ch[l + 1]
                g = nxt[1]
                for p, s in ((m.weight, c.view(-1, *[1] * (m.weight.dim() - 1))), (m.bias, c),
                             (nxt[0].weight, None)):
                    st = opt.state.get(p)
                    if not st:
                        continue
                    if s is None:  # outgoing weights were divided by c[g] along the input axis
                        s = (1.0 / c[g]).view(1, -1)
                        shape = st["exp_avg"].shape
                        st["exp_avg"].view(shape[0], -1).div_(s)
                        st["exp_avg_sq"].view(shape[0], -1).div_(s ** 2)
                    else:
                        st["exp_avg"].div_(s)
                        st["exp_avg_sq"].div_(s ** 2)


@torch.no_grad()
def imbalance(model):
    """Mean over units of |log(||in|| / ||out||)| per hidden layer: 0 = balanced."""
    ch = chain(model)
    out = []
    for l in range(len(ch) - 1):
        n_in, n_out = _in_out_norm2(ch, l)
        out.append(float(0.5 * (n_in / n_out).log().abs().mean()))
    return out


@torch.no_grad()
def conserved_quantity(model):
    """Per-unit ||in||^2 - ||out||^2, conserved exactly by gradient flow (Du et al., 2018)."""
    ch = chain(model)
    res = []
    for l in range(len(ch) - 1):
        n_in, n_out = _in_out_norm2(ch, l)
        res.append(n_in - n_out)
    return res
