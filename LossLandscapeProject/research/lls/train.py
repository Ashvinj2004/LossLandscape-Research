"""Training with loss-matched checkpoints.

A run trains until the full training-subset loss first drops below each threshold in
`targets`; at every such crossing the landscape measures are computed. This lets us compare
optimizers *at matched training loss*, removing the loss-level confound of fixed-epoch studies.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import measures as M
from .models import build

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


class SignSGD(torch.optim.Optimizer):
    """Plain sign descent: w <- w - lr * sign(g) (Bernstein et al., 2018)."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(torch.sign(p.grad), alpha=-group["lr"])


def make_optimizer(name, params, lr, wd=0.0):
    """wd: L2 weight decay for SGD-type methods; decoupled weight decay for AdamW."""
    if name == "signsgd":
        return SignSGD(params, lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.999)
    if name in ("sgd",):
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    if name in ("sgdm", "sam"):
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name in ("adam", "adamq"):
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def adam_preconditioner(opt, model):
    """Adam's effective diagonal preconditioner sqrt(v_hat) + eps, flattened."""
    parts = []
    for p in M.params_of(model):
        st = opt.state[p]
        b2 = opt.param_groups[0]["betas"][1]
        eps = opt.param_groups[0]["eps"]
        step = float(st["step"])
        vhat = st["exp_avg_sq"] / (1 - b2 ** step)
        parts.append((vhat.sqrt() + eps).reshape(-1))
    return torch.cat(parts)


def run(cfg, verbose=False):
    t0 = time.time()
    torch.set_num_threads(cfg.get("threads", 2))
    d = D.load(cfg["dataset"], n_train=cfg.get("n_train", 10000), standardize=cfg.get("standardize", True),
               n_eval=cfg.get("n_eval", 4096))
    xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    ev, te = (d["xev"], d["yev"]), (xte, yte)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build(cfg["arch"], cfg["dataset"])
    w_init = M.get_flat(model)
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"], cfg.get("wd", 0.0))
    rho = cfg.get("sam_rho", 0.05)
    targets = sorted(cfg.get("targets", [1.0, 0.3, 0.1, 0.03, 0.01]), reverse=True)
    max_epochs = cfg.get("max_epochs", 400)
    time_limit = cfg.get("time_limit", 3600)
    bs = cfg["bs"]
    n = len(ytr)
    g = torch.Generator().manual_seed(cfg["seed"])
    ckpts, history, traj = [], [], []
    from . import rescale as R
    track = not M.has_batchnorm(model)  # orbit bookkeeping assumes the plain ReLU symmetry
    cq0 = [c.clone() for c in R.conserved_quantity(model)] if track else None

    @torch.no_grad()
    def traj_point(epoch, step):
        cq = R.conserved_quantity(model)
        ch = R.chain(model)
        io = [R._in_out_norm2(ch, l) for l in range(len(ch) - 1)]
        return {"epoch": epoch, "step": step, "imbalance": R.imbalance(model),
                "cq_drift_mean": [float((c - c0).mean()) for c, c0 in zip(cq, cq0)],
                "cq_drift_absmean": [float((c - c0).abs().mean()) for c, c0 in zip(cq, cq0)],
                "in_norm2_mean": [float(a.mean()) for a, b in io],
                "out_norm2_mean": [float(b.mean()) for a, b in io]}

    if track:
        traj.append(traj_point(0, 0))
    tele_x = xtr[torch.randperm(n, generator=torch.Generator().manual_seed(99))[:1024]] if cfg.get("teleport_every") else None
    next_t = 0
    status = "max_epochs"
    step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):  # drop last incomplete batch
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            if cfg["opt"] == "sam":
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                ps = [p for p in model.parameters() if p.grad is not None]
                gn = torch.norm(torch.stack([p.grad.norm() for p in ps])) + 1e-12
                es = []
                with torch.no_grad():
                    for p in ps:
                        e = p.grad * (rho / gn)
                        p.add_(e)
                        es.append(e)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                with torch.no_grad():
                    for p, e in zip(ps, es):
                        p.sub_(e)
                opt.step()
            else:
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if cfg["opt"] == "adamq" and step % cfg.get("q_every", 1) == 0:
                    # intervention: undo Adam's drift along the rescaling orbit
                    R.restore_conserved(model, cq0, opt if cfg.get("q_transform", True) else None,
                                        layers=cfg.get("q_layers"))
            if cfg.get("teleport_every") and (step + 1) % cfg["teleport_every"] == 0:
                # jump to the minimum-sharpness point of the current orbit (function-preserving)
                if cfg.get("teleport_to", "min") == "balanced":  # minimum-norm point of the orbit
                    _, sc_t = R.balance_min_norm(model)
                else:  # minimum of sum_i diag(G)_ii^p over the orbit
                    dg_t = M.diag_ggn_linear(model, tele_x)
                    _, sc_t = R.orbit_min_trace(model, dg_t, p=cfg.get("teleport_p", 1.0))
                R.scale_units_inplace(model, sc_t, opt)
            step += 1
        model.eval()
        tr_loss, tr_acc = M.loss_acc(model, xtr, ytr)
        history.append((epoch, step, tr_loss, tr_acc))
        if track:
            traj.append(traj_point(epoch, step))
        if verbose:
            print(f"ep {epoch} step {step} loss {tr_loss:.4f} acc {tr_acc:.4f} t {time.time() - t0:.1f}s", flush=True)
        if not math.isfinite(tr_loss) or tr_loss > 50:
            status = "diverged"
            break
        crossed = []
        while next_t < len(targets) and tr_loss <= targets[next_t]:
            crossed.append(targets[next_t])
            next_t += 1
        if crossed and cfg.get("level", "full") == "none":
            te_loss, te_acc = M.loss_acc(model, xte, yte)
            ckpts.append({"target": crossed[-1], "epoch": epoch, "step": step, "train_loss": tr_loss,
                          "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc})
        elif crossed:
            te_loss, te_acc = M.loss_acc(model, xte, yte)
            pre = adam_preconditioner(opt, model) if cfg["opt"] in ("adam", "adamq", "adamw") else None
            tm = time.time()
            meas = M.all_measures(model, ev, te=te, w_init=w_init, precond=pre,
                                  level=cfg.get("level", "full"), seed=cfg.get("measure_seed", 0))
            ck = {"target": crossed[-1], "targets_crossed": crossed, "epoch": epoch, "step": step,
                  "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc,
                  "measure_time": time.time() - tm, **meas}
            ckpts.append(ck)
            if verbose:
                print(f"  ckpt target={crossed[-1]} test_acc={te_acc:.4f} lam={meas['lambda_max']:.2f} "
                      f"tr={meas['hess_trace']:.1f} ({time.time() - tm:.1f}s)", flush=True)
        if targets and next_t >= len(targets):
            status = "reached"
            break
        if time.time() - t0 > time_limit:
            status = "time_limit"
            break
    if status not in ("reached", "diverged") and cfg.get("level", "full") != "none":
        # Always measure the final model too, so every run has a terminal checkpoint.
        if not ckpts or ckpts[-1]["epoch"] != epoch:
            te_loss, te_acc = M.loss_acc(model, xte, yte)
            pre = adam_preconditioner(opt, model) if cfg["opt"] in ("adam", "adamq", "adamw") else None
            meas = M.all_measures(model, ev, te=te, w_init=w_init, precond=pre,
                                  level=cfg.get("level", "full"), seed=cfg.get("measure_seed", 0))
            tr_loss, tr_acc = M.loss_acc(model, xtr, ytr)
            ckpts.append({"target": None, "targets_crossed": [], "epoch": epoch, "step": step,
                          "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss,
                          "test_acc": te_acc, **meas})
    out = {"cfg": cfg, "status": status, "epochs": epoch, "steps": step, "time": time.time() - t0,
           "history": history, "traj": traj, "ckpts": ckpts}
    if cfg.get("save_model", True):
        os.makedirs(RUN_DIR, exist_ok=True)
        torch.save({"state": model.state_dict(), "cfg": cfg}, os.path.join(RUN_DIR, run_name(cfg) + ".pt"))
    return out


def run_name(cfg):
    extra = "_raw" if cfg.get("standardize", True) is False else ""
    extra += "_full" if cfg.get("n_train", 10000) is None else ""
    for k, tag in (("wd", "wd"), ("q_every", "qe"), ("q_transform", "qt"), ("q_layers", "ql"), ("teleport_every", "tele"), ("teleport_to", "to")):
        if k in cfg:
            v = cfg[k]
            extra += f"_{tag}{'-'.join(map(str, v)) if isinstance(v, list) else int(v) if isinstance(v, bool) else v}"
    return f"{cfg['dataset']}_{cfg['arch']}_{cfg['opt']}_lr{cfg['lr']}_bs{cfg['bs']}_s{cfg['seed']}{extra}"
