"""ResNet-18 on full CIFAR-10: raw vs rescaling-invariant sharpness at matched training loss (GPU).

Self-contained (needs only torch, numpy, scipy; torchvision only to download CIFAR-10), so it runs
unchanged on Colab or Kaggle. Measures follow lls/measures.py; see the paper's Sec. on BatchNorm for the
rationale. Every conv in ResNet-18 is followed by BatchNorm, so in batch-statistics mode the loss is
invariant to scaling each conv output channel. We report three tiers of measures:

  raw          tr H (Hutchinson), lambda_max (Lanczos), isotropic average sharpness, SAM worst-case,
               the original study's relative isotropic metric on the test set
  BN-invariant ntrace = tr H at the point where every conv filter has unit norm
               (sum over conv filters of ||w_f||^2 tr H_ff + the other diagonal entries; exact identity
               raw tr H = ntrace x (tr H / ntrace) splits a trace comparison into a BN-scale-invariant
               part and a part set by the filter norms), filter-normalized average sharpness
  invariant to every per-weight rescaling (BN scale, ReLU and residual-stream rescalings):
               sum_i w_i^2 H_ii, multiplicative average sharpness, ASAM worst-case

All measures use batch statistics (train-mode BN, running averages frozen) on a fixed set of clean
training images, in chunks of 128 (the training batch size), in float32.

Sweeps
  resnet18_noaug  the paper's protocol: no augmentation, constant learning rate, batch 128,
                  loss-matched checkpoints at clean training loss 1, 0.1, 0.01 (15 runs)
  resnet18_aug    a standard recipe: random crop + flip, weight decay, cosine schedule over 60 epochs;
                  checkpoints at the same losses plus the final model (9 runs)
  smoke           tiny CPU test of every code path

usage
  python scripts/resnet_cifar10.py --selftest                       # 1-2 min: correctness + timing
  python scripts/resnet_cifar10.py --sweep resnet18_noaug --out DIR [--shard 0/2] [--time_budget_h 11]
Runs are resumable: a finished run's JSON is skipped, an interrupted run continues from its state file.
"""
import argparse
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """The standard CIFAR-10 ResNet-18 (3x3 stem, no max-pool, [2,2,2,2] blocks, 11.2M parameters)."""

    def __init__(self, width=64, n_classes=10):
        super().__init__()
        w = width
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        layers, cin = [], w
        for cout, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.linear = nn.Linear(8 * w, n_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# ------------------------------------------------------------------------------------------ data
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10(data_dir):
    """uint8 arrays (N,3,32,32) and int64 labels, from cifar-10-batches-py (downloaded if missing)."""
    root = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(os.path.join(root, "test_batch")):
        os.makedirs(data_dir, exist_ok=True)
        try:
            import torchvision
            torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        except Exception as e:  # noqa: BLE001
            print("torchvision download failed (", e, "); trying direct download", flush=True)
            tgz = os.path.join(data_dir, "cifar-10-python.tar.gz")
            urllib.request.urlretrieve(CIFAR_URL, tgz)
            with tarfile.open(tgz) as t:
                t.extractall(data_dir)

    def read(name):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32), np.asarray(d[b"labels"], dtype=np.int64)

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = read("test_batch")
    reordered = os.path.exists(os.path.join(root, "REORDERED"))
    return np.concatenate(xs), np.concatenate(ys), xte, yte, reordered


class GPUData:
    """Whole dataset on the device; augmentation (random crop with 4px zero padding + horizontal flip,
    as torchvision's RandomCrop(32, 4) + RandomHorizontalFlip) is done per batch on the device."""

    def __init__(self, xtr, ytr, xte, yte, dev, n_eval, eval_seed=1235):
        self.dev = dev
        mean = xtr.reshape(len(xtr), 3, -1).mean(axis=(0, 2)) / 255.0
        std = (xtr.reshape(len(xtr), 3, -1) / 255.0).std(axis=(0, 2))
        self.mean = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        self.xtr_pad = F.pad(torch.from_numpy(xtr), (4, 4, 4, 4)).to(dev)  # uint8, N x 3 x 40 x 40
        self.ytr = torch.from_numpy(ytr).to(dev)
        self.xte = torch.from_numpy(xte).to(dev)
        self.yte = torch.from_numpy(yte).to(dev)
        self.n = len(ytr)
        ev = np.random.RandomState(eval_seed).choice(self.n, n_eval, replace=False)
        self.ev_idx = torch.from_numpy(np.sort(ev)).to(dev)
        self.xev = self.norm(self.xtr_pad[self.ev_idx, :, 4:36, 4:36])
        self.yev = self.ytr[self.ev_idx]
        self.base = torch.arange(32, device=dev)

    def norm(self, x_uint8):
        return ((x_uint8.float() / 255.0) - self.mean) / self.std

    def clean_train(self, idx):
        return self.norm(self.xtr_pad[idx, :, 4:36, 4:36])

    def augmented(self, idx, gen):
        B = idx.numel()
        oy = torch.randint(0, 9, (B,), generator=gen).to(self.dev)
        ox = torch.randint(0, 9, (B,), generator=gen).to(self.dev)
        flip = (torch.rand(B, generator=gen) < 0.5).to(self.dev)
        rows = oy[:, None] + self.base
        cols = ox[:, None] + self.base
        cols = torch.where(flip[:, None], cols.flip(1), cols)
        x = self.xtr_pad[idx]
        b = torch.arange(B, device=self.dev)[:, None, None]
        x = x[b, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)  # B x 3 x 32 x 32
        return self.norm(x)


# ------------------------------------------------------------------------------- measure helpers
def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]


def get_flat(model):
    return torch.cat([p.detach().reshape(-1) for p in params_of(model)]).clone()


@torch.no_grad()
def set_flat(model, v):
    i = 0
    for p in params_of(model):
        n = p.numel()
        p.copy_(v[i:i + n].view_as(p))
        i += n


def conv_filter_weights(model):
    """Flat vector (parameter order) with d_i = ||w_f||^2 for every conv weight in output filter f (every
    conv in ResNet-18 is followed by BatchNorm; conv weights are the only 4-D parameters) and d_i = 1 for
    every other parameter (BN affine, final linear layer)."""
    d = []
    for p in params_of(model):
        if p.dim() == 4:
            fn2 = p.detach().reshape(p.shape[0], -1).pow(2).sum(1)
            d.append(fn2[:, None].expand(p.shape[0], p[0].numel()).reshape(-1))
        else:
            d.append(torch.ones(p.numel(), device=p.device))
    return torch.cat(d)


class landscape_mode:
    """Batch statistics (train-mode BN) with running averages frozen (momentum 0)."""

    def __init__(self, model):
        self.model = model
        self.bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        self.was = self.model.training
        self.moms = [m.momentum for m in self.bns]
        self.model.train()
        for m in self.bns:
            m.momentum = 0.0
        return self

    def __exit__(self, *exc):
        for m, mo in zip(self.bns, self.moms):
            m.momentum = mo
        self.model.train(self.was)


def _chunks(X, Y, chunk):
    for i in range(0, len(Y), chunk):
        yield X[i:i + chunk], Y[i:i + chunk]


@torch.no_grad()
def loss_acc(model, X, Y, chunk, amp=False):
    tot, correct = 0.0, 0
    for xb, yb in _chunks(X, Y, chunk):
        with torch.autocast(X.device.type, dtype=torch.float16, enabled=amp and X.device.type == "cuda"):
            out = model(xb)
        tot += F.cross_entropy(out.float(), yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return tot / len(Y), correct / len(Y)


def grad_flat(model, X, Y, chunk):
    ps = params_of(model)
    g = [torch.zeros_like(p) for p in ps]
    N = len(Y)
    for xb, yb in _chunks(X, Y, chunk):
        loss = F.cross_entropy(model(xb), yb, reduction="sum") / N
        for a, b in zip(g, torch.autograd.grad(loss, ps)):
            a.add_(b)
    return torch.cat([a.reshape(-1) for a in g])


def hvp_flat(model, X, Y, v, chunk):
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
        for a, b in zip(out, torch.autograd.grad(dot, ps)):
            a.add_(b)
    return torch.cat([a.reshape(-1) for a in out])


def quad_traces(model, X, Y, chunk, n_probes, seed, scales):
    """Hutchinson estimates of tr(S^{1/2} H S^{1/2}) = sum_i s_i H_ii for each named scale vector s
    (None = identity), with Rademacher probes; returns {name: (mean, se)}."""
    w = get_flat(model)
    n = w.numel()
    out = {}
    for name, s in scales.items():
        g = torch.Generator().manual_seed(seed)
        r = None if s is None else s.sqrt()
        vals = []
        for _ in range(n_probes):
            z = (torch.randint(0, 2, (n,), generator=g).to(w.dtype) * 2 - 1).to(X.device)
            v = z if r is None else z * r
            vals.append(torch.dot(v, hvp_flat(model, X, Y, v, chunk)).item())
        vals = np.array(vals)
        out[name] = (float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals))))
    return out


def lambda_max(model, X, Y, chunk, iters=30, seed=0):
    """Largest Hessian eigenvalue by Lanczos with full reorthogonalization (bounded cost: `iters` HVPs).
    Returns (lambda_max, change of the estimate over the last 5 iterations, relative)."""
    w = get_flat(model)
    n = w.numel()
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=g, dtype=w.dtype).to(X.device)
    Q = [q / q.norm()]
    alphas, betas, est = [], [], []
    for j in range(iters):
        w = hvp_flat(model, X, Y, Q[-1], chunk)
        alphas.append(float(torch.dot(w, Q[-1])))
        for qq in Q:  # full reorthogonalization (twice is enough)
            w -= torch.dot(w, qq) * qq
        for qq in Q:
            w -= torch.dot(w, qq) * qq
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        est.append(float(np.linalg.eigvalsh(T)[-1]))
        b = float(w.norm())
        if b < 1e-8 * max(1.0, abs(est[-1])) or j == iters - 1:
            break
        betas.append(b)
        Q.append(w / b)
    drift = abs(est[-1] - est[max(0, len(est) - 6)]) / max(abs(est[-1]), 1e-12)
    return est[-1], drift


def perturbation(model, w0, kind, gen):
    eps = torch.randn(w0.numel(), generator=gen, dtype=w0.dtype).to(w0.device)
    if kind == "iso":
        return eps
    if kind == "mult":
        return eps * w0.abs()
    if kind == "filter":  # Li et al. 2018: per output filter / row, biases and BN parameters untouched
        d = torch.zeros_like(w0)
        i = 0
        for p in params_of(model):
            n = p.numel()
            if p.dim() > 1:
                e = eps[i:i + n].view(p.shape[0], -1)
                w = w0[i:i + n].view(p.shape[0], -1)
                d[i:i + n] = (e / (e.norm(dim=1, keepdim=True) + 1e-12) * w.norm(dim=1, keepdim=True)).reshape(-1)
            i += n
        return d
    raise ValueError(kind)


def average_sharpness(model, X, Y, chunk, kind, sigmas, n_draws, seed, L0):
    w0 = get_flat(model)
    out = {}
    for s in sigmas:
        g = torch.Generator().manual_seed(seed)  # common random numbers across sigmas and models
        incs = []
        for _ in range(n_draws):
            set_flat(model, w0 + s * perturbation(model, w0, kind, g))
            incs.append(loss_acc(model, X, Y, chunk)[0] - L0)
        set_flat(model, w0)
        incs = np.array(incs)
        out[s] = (float(incs.mean()), float(incs.std(ddof=1) / math.sqrt(n_draws)))
    return out


def worst_sharpness(model, X, Y, chunk, rho, adaptive, L0, steps=10):
    """max_{||u|| <= rho} L(w0 + T u) - L(w0), T = |w0| (ASAM) or I (SAM); normalized PGD as lls."""
    w0 = get_flat(model)
    T = w0.abs() if adaptive else torch.ones_like(w0)
    u = torch.zeros_like(w0)
    alpha = 4.0 * rho / steps
    best = 0.0
    for _ in range(steps):
        set_flat(model, w0 + T * u)
        g = grad_flat(model, X, Y, chunk) * T
        gn = g.norm()
        if gn < 1e-20:
            break
        u = u + alpha * g / gn
        un = u.norm()
        if un > rho:
            u = u * (rho / un)
        set_flat(model, w0 + T * u)
        best = max(best, loss_acc(model, X, Y, chunk)[0] - L0)
    set_flat(model, w0)
    return float(best)


@torch.no_grad()
def conv_norm_gmean(model):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    med = [float(m.weight.reshape(m.weight.shape[0], -1).norm(dim=1).median()) for m in convs]
    return float(np.exp(np.mean(np.log(med))))


@torch.no_grad()
def weight_stats(model, w_init):
    w0 = get_flat(model)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    med = [float(m.weight.reshape(m.weight.shape[0], -1).norm(dim=1).median()) for m in convs]
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    return {"w_norm": float(w0.norm()), "dist_init": float((w0 - w_init.to(w0.device)).norm()),
            "conv_filter_norm_median": med, "conv_filter_norm_gmean": float(np.exp(np.mean(np.log(med)))),
            "bn_gamma_absmean": float(torch.cat([b.weight.abs() for b in bns]).mean()),
            "fc_norm": float(model.linear.weight.norm())}


SIGMAS = {"iso": [3e-4, 1e-3, 3e-3], "filter": [0.01, 0.03, 0.1], "mult": [0.01, 0.03, 0.1]}


def measure_all(model, data, w_init, args):
    X, Y, chunk = data.xev, data.yev, args.chunk
    t0 = time.time()
    r = weight_stats(model, w_init)
    with landscape_mode(model):
        L0, A0 = loss_acc(model, X, Y, chunk)
        r.update({"ev_loss": L0, "ev_acc": A0})
        d = conv_filter_weights(model)
        w = get_flat(model)
        tr = quad_traces(model, X, Y, chunk, args.n_probes, seed=0,
                         scales={"hess_trace": None, "wtrace": w * w, "ntrace": d})
        for k, (m, se) in tr.items():
            r[k], r[k + "_se"] = m, se
        r["lambda_max"], r["lambda_max_conv"] = lambda_max(model, X, Y, chunk)
        for kind, sig in SIGMAS.items():
            for s, (m, se) in average_sharpness(model, X, Y, chunk, kind, sig, args.n_draws, 0, L0).items():
                r[f"avg_{kind}_{s}"], r[f"avg_{kind}_{s}_se"] = m, se
        r["worst_sam_0.05"] = worst_sharpness(model, X, Y, chunk, 0.05, False, L0)
        r["worst_asam_0.5"] = worst_sharpness(model, X, Y, chunk, 0.5, True, L0)
        # the original study's metric: isotropic sigma = 0.01, relative, on the test set, 10 draws
        Xt = data.norm(data.xte)
        Lt = loss_acc(model, Xt, data.yte, chunk)[0]
        for s in [0.01, 0.001]:
            inc = average_sharpness(model, Xt, data.yte, chunk, "iso", [s], 10, 7, Lt)[s][0]
            r["orig_rel_sharp_test" if s == 0.01 else f"rel_iso_test_{s}"] = inc / Lt
        r["ev_test_loss_batchstat"] = Lt
    r["measure_time"] = time.time() - t0
    return r


# ------------------------------------------------------------------------------------- training
def make_optimizer(name, params, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def make_scaler(dev, enabled):
    if dev.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_name(c):
    return f"{c['sweep']}_{c['opt']}_lr{c['lr']:g}_wd{c['wd']:g}_s{c['seed']}"


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def atomic_torch(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def train_run(cfg, data, args, out_dir, deadline):
    """Returns 'done' or 'stopped' (time budget reached; state saved for resuming)."""
    dev = data.dev
    name = run_name(cfg)
    jpath = os.path.join(out_dir, name + ".json")
    spath = os.path.join(out_dir, name + ".state.pt")
    torch.manual_seed(cfg["seed"])
    model = ResNet18(width=cfg["width"]).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"], cfg["wd"])
    steps_per_epoch = data.n // cfg["bs"]
    total_steps = steps_per_epoch * cfg["max_epochs"]
    if cfg["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, total_steps) / total_steps)))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0)
    amp = args.amp and dev.type == "cuda"
    scaler = make_scaler(dev, amp)
    gen = torch.Generator().manual_seed(cfg["seed"])
    w_init = get_flat(model).cpu()
    targets = sorted(cfg["targets"], reverse=True)
    st = {"epoch": 0, "step": 0, "history": [], "ckpts": [], "next_t": 0, "train_time": 0.0}
    if os.path.exists(spath):
        S = torch.load(spath, map_location="cpu", weights_only=False)
        model.load_state_dict(S["model"])
        opt.load_state_dict(S["opt"])
        sched.load_state_dict(S["sched"])
        if scaler is not None and S.get("scaler"):
            scaler.load_state_dict(S["scaler"])
        gen.set_state(S["gen"])
        w_init = S["w_init"]
        st = S["st"]
        print(f"  resumed {name} at epoch {st['epoch']}", flush=True)

    def save_state():
        atomic_torch({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict() if scaler is not None else None, "gen": gen.get_state(),
                      "w_init": w_init, "st": st}, spath)

    def result(status):
        return {"cfg": cfg, "status": status, "epochs": st["epoch"], "steps": st["step"],
                "time": st["train_time"], "history": st["history"], "ckpts": st["ckpts"]}

    def checkpoint(target, crossed, tr, te):
        model.eval()
        meas = measure_all(model, data, w_init, args)
        ck = {"target": target, "targets_crossed": crossed, "epoch": st["epoch"], "step": st["step"],
              "lr": opt.param_groups[0]["lr"], "train_loss": tr[0], "train_acc": tr[1],
              "test_loss": te[0], "test_acc": te[1], **meas}
        st["ckpts"].append(ck)
        print(f"  ckpt {name} target={target} ep {st['epoch']} trH {meas['hess_trace']:.1f} "
              f"wtrace {meas['wtrace']:.3f} ntrace {meas['ntrace']:.1f} lam {meas['lambda_max']:.2f} "
              f"({meas['measure_time']:.0f}s)", flush=True)
        atomic_json(result("partial"), os.path.join(out_dir, name + ".partial.json"))

    def evaluate():
        model.eval()
        Xall = data.xtr_pad[:, :, 4:36, 4:36]
        tot, corr = 0.0, 0
        with torch.no_grad():
            for j in range(0, data.n, 1000):
                xb = data.norm(Xall[j:j + 1000])
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    out = model(xb)
                tot += F.cross_entropy(out.float(), data.ytr[j:j + 1000], reduction="sum").item()
                corr += (out.argmax(1) == data.ytr[j:j + 1000]).sum().item()
        return (tot / data.n, corr / data.n), loss_acc(model, data.norm(data.xte), data.yte, 1000, amp=amp)

    status = "max_epochs"
    while st["epoch"] < cfg["max_epochs"]:
        if time.time() > deadline:
            save_state()
            return "stopped"
        t0 = time.time()
        model.train()
        perm = torch.randperm(data.n, generator=gen).to(dev)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg["bs"]:(i + 1) * cfg["bs"]]
            xb = data.augmented(idx, gen) if cfg["augment"] else data.clean_train(idx)
            yb = data.ytr[idx]
            if dev.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            st["step"] += 1
        st["epoch"] += 1
        # clean training loss (eval-mode BN, the network as it would be deployed) decides the targets
        tr, te = evaluate()
        st["train_time"] += time.time() - t0
        st["history"].append([st["epoch"], st["step"], tr[0], tr[1], te[0], te[1], opt.param_groups[0]["lr"],
                              conv_norm_gmean(model)])
        print(f"{name} ep {st['epoch']} loss {tr[0]:.4f} acc {tr[1]:.4f} test {te[1]:.4f} "
              f"lr {opt.param_groups[0]['lr']:.2e} ({time.time() - t0:.0f}s)", flush=True)
        if not math.isfinite(tr[0]) or tr[0] > 50:
            status = "diverged"
            break
        crossed = []
        while st["next_t"] < len(targets) and tr[0] <= targets[st["next_t"]]:
            crossed.append(targets[st["next_t"]])
            st["next_t"] += 1
        if crossed:
            checkpoint(crossed[-1], crossed, tr, te)
        if st["next_t"] >= len(targets) and not cfg["run_to_end"]:
            status = "reached"
            break
        if crossed or st["epoch"] % args.save_every == 0:
            save_state()
    if status != "diverged" and (cfg["run_to_end"] or status == "max_epochs"):
        if not st["ckpts"] or st["ckpts"][-1]["epoch"] != st["epoch"]:
            tr, te = evaluate()
            checkpoint(None, [], tr, te)
    res = result(status)
    res["git_commit"] = git_commit()
    res["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    atomic_json(res, jpath)
    for p in (spath, os.path.join(out_dir, name + ".partial.json")):
        if os.path.exists(p):
            os.remove(p)
    print(f"DONE {name} {status} epochs {st['epoch']} train {st['train_time'] / 60:.1f} min", flush=True)
    return "done"


# --------------------------------------------------------------------------------------- sweeps
def sweep_configs(sweep):
    cfgs = []
    if sweep == "resnet18_noaug":
        # the paper's protocol: clean data, constant learning rate, loss-matched checkpoints
        opts = [("sgdm", 0.01), ("sgdm", 0.05), ("adam", 1e-4), ("adam", 5e-4), ("sgd", 0.1)]
        for seed, (opt, lr) in itertools.product([0, 1, 2], opts):
            cfgs.append(dict(sweep=sweep, opt=opt, lr=lr, wd=0.0, seed=seed, bs=128, width=64, augment=False,
                             schedule="const", max_epochs=60, targets=[1.0, 0.1, 0.01], run_to_end=False))
    elif sweep == "resnet18_aug":
        # a standard recipe: augmentation, weight decay, cosine schedule; AdamW's decoupled 0.05 x 1e-3
        # equals SGD+M's 5e-4 x 0.1 per-step shrinkage
        opts = [("sgdm", 0.1, 5e-4), ("adam", 1e-3, 5e-4), ("adamw", 1e-3, 0.05)]
        for seed, (opt, lr, wd) in itertools.product([0, 1, 2], opts):
            cfgs.append(dict(sweep=sweep, opt=opt, lr=lr, wd=wd, seed=seed, bs=128, width=64, augment=True,
                             schedule="cosine", max_epochs=60, targets=[1.0, 0.1, 0.01], run_to_end=True))
    elif sweep == "smoke":
        for seed, (opt, lr, aug) in itertools.product([0], [("sgdm", 0.05, False), ("adam", 1e-3, True)]):
            cfgs.append(dict(sweep=sweep, opt=opt, lr=lr, wd=5e-4 if aug else 0.0, seed=seed, bs=32, width=4,
                             augment=aug, schedule="cosine" if aug else "const", max_epochs=3,
                             targets=[2.2, 2.0], run_to_end=aug))
    else:
        raise ValueError(sweep)
    return cfgs


# -------------------------------------------------------------------------------------- selftest
def selftest(args, dev):
    """Correctness checks on the target device, then (optionally) a timing estimate for the sweeps."""
    ok = True
    torch.manual_seed(0)
    model = ResNet18(width=args.width_test).to(dev)
    X = torch.randn(64, 3, 32, 32, device=dev)
    Y = torch.randint(0, 10, (64,), device=dev)
    chunk = 32
    rel = lambda a, b: abs(a - b) / max(abs(a), 1e-12)  # noqa: E731
    checks = [("filter-norm vector aligned with parameters", conv_filter_weights(model).numel() == get_flat(model).numel())]

    def snapshot():
        L = loss_acc(model, X, Y, chunk)[0]
        w = get_flat(model)
        q = quad_traces(model, X, Y, chunk, 4, 0, {"tr": None, "wtr": w * w, "ntr": conv_filter_weights(model)})
        return {"loss": L, "tr": q["tr"][0], "wtr": q["wtr"][0], "ntr": q["ntr"][0],
                "mult": average_sharpness(model, X, Y, chunk, "mult", [0.03], 4, 0, L)[0.03][0],
                "asam": worst_sharpness(model, X, Y, chunk, 0.5, True, L, steps=3)}

    gs = torch.Generator().manual_seed(2)
    with landscape_mode(model):
        a = snapshot()
        with torch.no_grad():  # BN scale symmetry: scale every conv's output filters
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.mul_(torch.exp(0.5 * torch.randn(m.weight.shape[0], generator=gs)).to(dev).view(-1, 1, 1, 1))
        b = snapshot()
        checks += [("BN filter scaling: loss unchanged", rel(a["loss"], b["loss"]) < 1e-3),
                   ("BN filter scaling: raw trace changes", rel(a["tr"], b["tr"]) > 5e-2),
                   ("BN filter scaling: normalized trace unchanged", rel(a["ntr"], b["ntr"]) < 1e-2),
                   ("BN filter scaling: sum w^2 H_ii unchanged", rel(a["wtr"], b["wtr"]) < 1e-2),
                   ("BN filter scaling: multiplicative sharpness unchanged", rel(a["mult"], b["mult"]) < 1e-2),
                   ("BN filter scaling: ASAM unchanged", rel(a["asam"], b["asam"]) < 1e-2)]
        with torch.no_grad():  # ReLU rescaling inside a block: BN1 affine x s, next conv's inputs / s
            blk = model.layer3[1]
            s = torch.exp(0.5 * torch.randn(blk.bn1.weight.shape[0], generator=gs)).to(dev)
            blk.bn1.weight.mul_(s)
            blk.bn1.bias.mul_(s)
            blk.conv2.weight.div_(s.view(1, -1, 1, 1))
        c = snapshot()
        checks += [("ReLU rescaling: loss unchanged", rel(b["loss"], c["loss"]) < 1e-4),
                   ("ReLU rescaling: sum w^2 H_ii unchanged", rel(b["wtr"], c["wtr"]) < 1e-2),
                   ("ReLU rescaling: multiplicative sharpness unchanged", rel(b["mult"], c["mult"]) < 1e-2),
                   ("ReLU rescaling: ASAM unchanged", rel(b["asam"], c["asam"]) < 1e-2)]
        # Lanczos against the Rayleigh quotient bound: lambda_max >= v^T H v for any unit v
        g = torch.Generator().manual_seed(1)
        v = torch.randn(get_flat(model).numel(), generator=g).to(dev)
        v /= v.norm()
        hv = hvp_flat(model, X, Y, v, chunk)
        lam, _ = lambda_max(model, X, Y, chunk, iters=20)
        checks.append(("Lanczos estimate >= Rayleigh quotient", lam >= float(torch.dot(v, hv)) - 1e-6))
    # HVP against a central finite difference of gradients, in float64 and with a step small enough
    # not to flip any ReLU (a flipped ReLU makes the finite difference meaningless, not the HVP wrong)
    m64 = ResNet18(width=args.width_test).to(dev).double()
    X64, eps = X[:32].double(), 1e-6
    with landscape_mode(m64):
        v = torch.randn(get_flat(m64).numel(), generator=g, dtype=torch.float64).to(dev)
        v /= v.norm()
        hv = hvp_flat(m64, X64, Y[:32], v, 16)
        w0 = get_flat(m64)
        set_flat(m64, w0 + eps * v)
        gp = grad_flat(m64, X64, Y[:32], 16)
        set_flat(m64, w0 - eps * v)
        gm = grad_flat(m64, X64, Y[:32], 16)
        set_flat(m64, w0)
        fd = (gp - gm) / (2 * eps)
        checks.append(("HVP matches finite differences (float64)", float((hv - fd).norm() / fd.norm()) < 1e-4))
    for nm, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {nm}", flush=True)
        ok &= bool(good)
    # 3. timing at full width on the real data pipeline
    if args.time_estimate:
        xtr, ytr, xte, yte, _ = load_cifar10(args.data)
        data = GPUData(xtr, ytr, xte, yte, dev, args.n_eval)
        model = ResNet18(width=64).to(dev)
        if dev.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        amp = args.amp and dev.type == "cuda"
        scaler = make_scaler(dev, amp)
        gen = torch.Generator().manual_seed(0)
        model.train()
        nsteps = 30 if dev.type == "cuda" else 3
        for i in range(nsteps + 5):
            if i == 5:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            idx = torch.randint(0, data.n, (128,), generator=gen).to(dev)
            xb = data.augmented(idx, gen)
            if dev.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), data.ytr[idx])
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_step = (time.time() - t0) / nsteps
        t_epoch = t_step * (data.n // 128) * 1.15  # + per-epoch evaluation
        with landscape_mode(model):
            v = torch.randn(get_flat(model).numel(), generator=gen).to(dev)
            t0 = time.time()
            hvp_flat(model, data.xev, data.yev, v, args.chunk)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_hvp = time.time() - t0
        t_meas = t_hvp * (3 * args.n_probes + 25) + 120  # traces + Lanczos + perturbation measures
        h_noaug = 15 * (25 * t_epoch + 3 * t_meas) / 3600
        h_aug = 9 * (60 * t_epoch + 4 * t_meas) / 3600
        print(f"  timing: {t_step * 1000:.0f} ms/step, ~{t_epoch:.0f} s/epoch, {t_hvp:.1f} s/HVP, "
              f"~{t_meas / 60:.1f} min per measurement", flush=True)
        print(f"  estimated GPU time: resnet18_noaug ~{h_noaug:.1f} h, resnet18_aug ~{h_aug:.1f} h "
              f"(on this device; halve the wall time with two GPUs)", flush=True)
        if dev.type == "cuda":
            print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
    return ok


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default="resnet_results")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--time_budget_h", type=float, default=1e9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--n_probes", type=int, default=16)
    ap.add_argument("--n_draws", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--n_train", type=int, default=None, help="testing only: use a training subset")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width_test", type=int, default=8)
    ap.add_argument("--time_estimate", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # float32 measurements must be float32 (TF32 on Ampere+ would cost ~3 digits in HVPs);
        # training uses fp16 autocast and is unaffected
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("device:", torch.cuda.get_device_name(dev), flush=True)
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    if args.selftest:
        sys.exit(0 if selftest(args, dev) else 1)
    start = time.time()
    deadline = start + args.time_budget_h * 3600
    k, n = (int(t) for t in args.shard.split("/"))
    cfgs = [c for i, c in enumerate(sweep_configs(args.sweep)) if i % n == k]
    out_dir = os.path.join(args.out, args.sweep)
    os.makedirs(out_dir, exist_ok=True)
    xtr, ytr, xte, yte, reordered = load_cifar10(args.data)
    if args.n_train:
        sub = np.random.RandomState(0).choice(len(ytr), args.n_train, replace=False)
        xtr, ytr, xte, yte = xtr[sub], ytr[sub], xte[:args.n_train], yte[:args.n_train]
    data = GPUData(xtr, ytr, xte, yte, dev, min(args.n_eval, len(ytr)))
    todo = [c for c in cfgs if not os.path.exists(os.path.join(out_dir, run_name(c) + ".json"))]
    print(f"{args.sweep} shard {args.shard}: {len(todo)} of {len(cfgs)} runs to do "
          f"(train set {data.n}{', REORDERED copy' if reordered else ''})", flush=True)
    for i, c in enumerate(todo, 1):
        c = dict(c, n_train=data.n, n_eval=len(data.yev), chunk=args.chunk, n_probes=args.n_probes,
                 n_draws=args.n_draws, amp=args.amp, data_reordered=reordered)
        print(f"[{i}/{len(todo)}] {run_name(c)}", flush=True)
        if train_run(c, data, args, out_dir, deadline) == "stopped":
            print(f"TIME BUDGET REACHED after {(time.time() - start) / 3600:.1f} h: state saved; "
                  f"run again to resume", flush=True)
            return
    print(f"ALL DONE {args.sweep} shard {args.shard} in {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
