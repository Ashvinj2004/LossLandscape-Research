"""Paper figures. usage: python scripts/make_figures.py  (writes research/figures/*.pdf and *.png)

Colour = optimizer identity, fixed order, validated (dataviz validator, light mode: all checks pass;
green/red adjacent pair is in the CVD warn band, so every optimizer also has its own marker shape).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lls.analysis import load_sweep  # noqa: E402
from analyze_main import MEASURES, prob_flatter  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

STYLE = {  # optimizer -> (label, colour, marker, linestyle)
    "adam": ("Adam", "#2a78d6", "o", "-"),
    "sgd": ("SGD", "#eb6834", "s", "-"),
    "sgdm": ("SGD+M", "#4a3aa7", "^", "-"),
    "sam": ("SAM", "#008300", "D", "-"),
    "adamq": ("Adam-Q", "#e34948", "X", "--"),
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DS_LABEL = {"main_cifar10_mlp": "CIFAR-10 MLP", "main_mnist_mlp": "MNIST MLP", "cnn_cifar10": "CIFAR-10 CNN"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def sweep_df(name):
    if not glob.glob(os.path.join(RES, name, "*.json")):
        return None
    df = load_sweep(name)
    return df[df["target"].notna()]


# ---------------------------------------------------------------- Figure 1: the ordering flip
FLIP_MEASURES = [  # (display, column, family)
    ("Original metric: rel. iso. ΔL/L (test)", "orig_rel_sharp_test", "raw"),
    ("Isotropic avg. sharpness", "avg_iso_0.01", "raw"),
    ("Hessian trace", "hess_trace", "raw"),
    ("Σ √diag G  (Adam's implicit reg.)", "ggn_sqrt_trace", "raw"),
    ("λ_max", "lambda_max", "raw"),
    ("Minimum sharpness (min. trace over orbit)", "orbitmin_trace", "inv"),
    ("Orbit-min Σ √diag G", "orbitmin_sqrt_trace", "inv"),
    ("Filter-normalized avg. sharpness", "avg_filter_0.03", "inv"),
    ("Multiplicative avg. sharpness", "avg_mult_0.03", "inv"),
    ("Σ w² G_ii", "ggn_trace_wscaled", "inv"),
]


def fig_flip(sweeps, target=0.1, other="sgd", name="fig1_flip"):
    dfs = {s: sweep_df(s) for s in sweeps}
    dfs = {s: d for s, d in dfs.items() if d is not None and (d["opt"] == "adam").any()}
    if not dfs:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = np.arange(len(FLIP_MEASURES))[::-1]
    ax.axvspan(0.5, 1.0, color="#f0efec", zorder=0)
    ax.axvline(0.5, color="#c3c2b7", lw=0.8)
    markers = ["o", "s", "^"]
    for k, (s, d) in enumerate(dfs.items()):
        vals = []
        for disp, col, fam in FLIP_MEASURES:
            p, _ = prob_flatter(d, col, "adam", other, target) if col in d else (np.nan, 0)
            vals.append(p)
        ax.plot(vals, ys + (k - (len(dfs) - 1) / 2) * 0.22, ls="none", marker=markers[k], ms=5,
                color=["#2a78d6", "#eb6834", "#4a3aa7"][k], label=DS_LABEL.get(s, s),
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in FLIP_MEASURES], color=INK2)
    n_raw = sum(1 for m in FLIP_MEASURES if m[2] == "raw")
    ax.axhline(ys[n_raw - 1] - 0.5, color="#c3c2b7", lw=0.8)
    ax.text(1.02, ys[0], "depends on\ncoordinates", transform=ax.get_yaxis_transform(), fontsize=6.5,
            color=MUTED, va="top")
    ax.text(1.02, ys[n_raw], "rescaling-\ninvariant", transform=ax.get_yaxis_transform(), fontsize=6.5,
            color=MUTED, va="top")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"P(Adam's minimum flatter than {STYLE[other][0]}'s)  at train loss {target}")
    ax.text(0.97, ys[0] + 0.9, "Adam looks flatter →", ha="right", fontsize=6.2, color=INK2)
    ax.text(0.03, ys[0] + 0.9, "← Adam looks sharper", ha="left", fontsize=6.2, color=INK2)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, -0.02), handletextpad=0.2, fontsize=6.5)
    ax.set_title("(a) The optimizer ordering flips with the measure", loc="left")

    # (b) exact decomposition of the raw trace comparison: log2(Adam/X) = function part + coordinate part
    width = 0.36
    ticks, ticklabels = [], []
    offset = 0
    for k, (s_, d) in enumerate(dfs.items()):
        targets = sorted(d["target"].unique(), reverse=True)
        d = d.assign(l_raw=np.log2(d["ggn_trace"]), l_fun=np.log2(d["orbitmin_trace"]))
        d = d.assign(l_coord=d["l_raw"] - d["l_fun"])
        xs = np.arange(len(targets)) + offset
        offset += len(targets) + 1
        fun, coord, raw = [], [], []
        for t in targets:
            a, x = d[(d.opt == "adam") & (d.target == t)], d[(d.opt == other) & (d.target == t)]
            fun.append(a.l_fun.mean() - x.l_fun.mean())
            coord.append(a.l_coord.mean() - x.l_coord.mean())
            raw.append(a.l_raw.mean() - x.l_raw.mean())
        bx.bar(xs - width / 2, fun, width=width, color="#1c5cab", label="function (minimum sharpness)" if k == 0 else None)
        bx.bar(xs + width / 2, coord, width=width, color="#86b6ef", label="coordinates (orbit excess)" if k == 0 else None)
        bx.plot(xs, raw, ls="none", marker="D", ms=5, color=INK, markeredgecolor="white", markeredgewidth=0.6,
                label="raw trace = sum" if k == 0 else None)
        bx.text(xs.mean(), -1.22, DS_LABEL.get(s_, s_), ha="center", fontsize=6.8, color=INK2)
        ticks += list(xs)
        ticklabels += [f"{t:g}".replace("0.", ".") for t in targets]
    bx.axhline(0, color="#c3c2b7", lw=0.8)
    bx.set_xticks(ticks)
    bx.set_xticklabels(ticklabels, fontsize=5.8)
    bx.set_xlabel("training loss at which the comparison is made", fontsize=7)
    bx.set_ylim(-1.3, 1.7)
    bx.set_ylabel(f"log₂(Adam / {STYLE[other][0]}) of Hessian trace\n← Adam flatter      Adam sharper →", fontsize=7)
    bx.legend(loc="upper right", fontsize=6.2, handlelength=1.2)
    bx.grid(axis="x", visible=False)
    bx.set_title("(b) Raw trace = function × coordinates (exact)", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


INIT_EXCESS = {"main_cifar10_mlp": 3.66, "main_mnist_mlp": 1.79}


# ---------------------------------------------------------------- Figure 2: mechanism
def load_drift():
    rows, logs = [], {}
    for p in glob.glob(os.path.join(RES, "drift", "*.json")):
        r = json.load(open(p))
        c = r["cfg"]
        logs[(c["opt"], c["lr"], c["res"], c["seed"])] = r["log"]
        last = r["log"][-1]
        rows.append(dict(opt=c["opt"], lr=c["lr"], res=c["res"], seed=c["seed"], d_in=r["d_in"],
                         dq1=last["first_order_mean"][0], dq2=last["second_order_mean"][0],
                         excess=r["ggn_trace"] / r["orbitmin_trace"]))
    return pd.DataFrame(rows), logs


def fig_mechanism(name="fig2_mechanism"):
    df, logs = load_drift()
    if df.empty:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35))
    # (a) decomposition over steps
    ax = axs[0]
    for (opt, lr), lab, c, mk in [(("adam", 0.0003), "Adam", STYLE["adam"][1], "o"),
                                  (("sgd", 0.03), "SGD", STYLE["sgd"][1], "s")]:
        lg = logs.get((opt, lr, 32, 0))
        if lg is None:
            continue
        st = np.array([e["step"] for e in lg])
        f1 = np.abs([e["first_order_mean"][0] for e in lg])
        f2 = np.abs([e["second_order_mean"][0] for e in lg])
        ax.plot(st, np.maximum(f1, 1e-10), color=c, ls="-", marker=mk, ms=3, markevery=4,
                label=f"{lab}: 1st-order term")
        ax.plot(st, np.maximum(f2, 1e-10), color=c, ls=":", label=f"{lab}: 2nd-order term")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 10)
    ax.set_xlabel("step")
    ax.set_ylabel("|cumulative drift of Q|  (layer 1)")
    ax.set_title("(a) Exact drift decomposition", loc="left")
    ax.legend(loc="center right", fontsize=6, handlelength=1.6)
    # (b) scaling law for Adam
    bx = axs[1]
    a = df[df.opt == "adam"]
    x = a.lr * np.sqrt(a.d_in)
    for res, mk, c in [(8, "o", "#86b6ef"), (16, "s", "#3987e5"), (32, "^", "#1c5cab")]:
        s = a[a.res == res]
        bx.plot(s.lr * np.sqrt(s.d_in), s.dq1, ls="none", marker=mk, color=c, ms=5,
                markeredgecolor="white", markeredgewidth=0.6, label=f"d_in = {int(s.d_in.iloc[0])}")
    lo = a[a.lr <= 1e-4]
    coef = np.polyfit(np.log(lo.lr * np.sqrt(lo.d_in)), np.log(lo.dq1), 1)
    xs = np.geomspace(x.min(), x.max(), 50)
    bx.plot(xs, np.exp(np.polyval(coef, np.log(xs))), color=MUTED, lw=1, ls="--")
    bx.text(0.04, 0.92, f"slope {coef[0]:.2f} (fit on η ≤ 1e-4)", transform=bx.transAxes, fontsize=6.5, color=INK2)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("η · √fan-in")
    bx.set_ylabel("Adam 1st-order drift (layer 1)")
    bx.set_title("(b) Adam's drift ∝ η √fan-in", loc="left")
    bx.legend(loc="lower right", fontsize=6, handletextpad=0.2)
    # (c) orbit excess at matched loss, both datasets
    cx = axs[2]
    for s_, ls_ in [("main_cifar10_mlp", "-"), ("main_mnist_mlp", "--")]:
        d = sweep_df(s_)
        if d is None:
            continue
        d = d.assign(excess=d["ggn_trace"] / d["orbitmin_trace"])
        for opt in ["sgdm", "sgd", "sam", "adam"]:
            med = d[d["opt"] == opt].groupby("target")["excess"].median()
            lab, c, mk, _ = STYLE[opt]
            cx.plot(med.index, med.values, color=c, marker=mk, ls=ls_, ms=3.5,
                    label=lab if s_ == "main_cifar10_mlp" else None, markeredgecolor="white", markeredgewidth=0.5)
    cx.plot([], [], color=MUTED, ls="-", label="CIFAR-10")
    cx.plot([], [], color=MUTED, ls="--", label="MNIST")
    cx.axhline(1.0, color="#c3c2b7", lw=0.8)
    cx.set_xscale("log")
    cx.invert_xaxis()
    cx.set_xlim(1.3, 0.007)
    cx.set_ylim(0.9, 4.1)
    cx.set_xlabel("training loss")
    cx.set_ylabel("orbit excess  tr G / min. sharpness")
    cx.set_title("(c) Adam pins its orbit excess", loc="left")
    cx.legend(loc="upper left", fontsize=5.8, ncol=2, handletextpad=0.3, columnspacing=0.6, handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, name)


# ---------------------------------------------------------------- Figure 3: intervention
def fig_intervention(pairs=(("main_cifar10_mlp", "interv_cifar10"), ("main_mnist_mlp", "interv_mnist")),
                     name="fig3_intervention"):
    cols = [("Orbit excess  tr G / MS₁", "excess", "coord"), ("Hessian trace", "hess_trace", "raw"),
            ("Isotropic avg.", "avg_iso_0.01", "raw"), ("Original metric", "orig_rel_sharp_test", "raw"),
            ("λ_max", "lambda_max", "raw"), ("Minimum sharpness", "orbitmin_trace", "inv"),
            ("Filter-norm. avg.", "avg_filter_0.1", "inv"), ("Multiplicative avg.", "avg_mult_0.1", "inv"),
            ("Σ w² G_ii", "ggn_trace_wscaled", "inv")]
    have = []
    for main, interv in pairs:
        dm, di = sweep_df(main), sweep_df(interv)
        if dm is None or di is None:
            continue
        key = ["lr", "bs", "seed", "target"]
        dm = dm.assign(excess=dm["ggn_trace"] / dm["orbitmin_trace"])
        di = di.assign(excess=di["ggn_trace"] / di["orbitmin_trace"])
        m = dm[dm.opt == "adam"].merge(di, on=key, suffixes=("_adam", "_q"))
        # keep only pairs whose checkpoints really are loss-matched (both within 25% of each other)
        m = m[(np.log(m["train_loss_q"] / m["train_loss_adam"])).abs() < np.log(1.25)]
        have.append((main, m))
    if not have:
        return
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ys = np.arange(len(cols))[::-1]
    for k, (main, m) in enumerate(have):
        for j, (disp, col, fam) in enumerate(cols):
            r = np.log2(m[f"{col}_q"] / m[f"{col}_adam"]).replace([np.inf, -np.inf], np.nan).dropna()
            med = r.median()
            lo_, hi = r.quantile(0.25), r.quantile(0.75)
            y = ys[j] + (k - 0.5) * 0.25
            ax.plot([lo_, hi], [y, y], color=["#2a78d6", "#eb6834"][k], lw=1.2)
            ax.plot(med, y, marker=["o", "s"][k], color=["#2a78d6", "#eb6834"][k], ms=5,
                    markeredgecolor="white", markeredgewidth=0.6, label=DS_LABEL[main] if j == 0 else None)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.axhline(ys[0] - 0.5, color="#c3c2b7", lw=0.8)
    ax.axhline(ys[4] - 0.5, color="#c3c2b7", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in cols], color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("log₂( Adam-Q / Adam )  at matched loss  (median, IQR)")
    ax.set_title("Removing Adam's symmetry drift: raw sharpness rises, invariant sharpness does not", loc="left",
                 fontsize=7.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- Figure 4: replication of the original study
def fig_replication(name="fig4_replication"):
    if not glob.glob(os.path.join(RES, "orig_replication", "*.json")):
        return
    df = load_sweep("orig_replication")
    cols = [("Original metric\nrel. iso ΔL/L (test)", "orig_rel_sharp_test"), ("Test loss", "test_loss"),
            ("Hessian trace", "hess_trace"), ("Minimum sharpness", "orbitmin_trace"),
            ("Filter-norm. avg.", "avg_filter_0.03")]
    fig, axs = plt.subplots(1, len(cols), figsize=(7.0, 1.9))
    for ax, (disp, col) in zip(axs, cols):
        for i, ds in enumerate(["mnist", "cifar10"]):
            for j, opt in enumerate(["sgd", "adam"]):
                v = df[(df.dataset == ds) & (df.opt == opt)][col]
                x = i * 2.6 + j
                lab, c, mk, _ = STYLE[opt]
                ax.bar(x, v.mean(), width=0.8, color=c, alpha=0.9, label=lab if (i == 0 and ax is axs[0]) else None)
                ax.plot(np.full(len(v), x), v, ls="none", marker=mk, color="white", ms=3, markeredgecolor=INK2,
                        markeredgewidth=0.5)
        ax.set_xticks([0.5, 3.1])
        ax.set_xticklabels(["MNIST", "CIFAR-10"])
        ax.set_title(disp, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_yscale("log" if col in ("hess_trace", "orbitmin_trace") else "linear")
    axs[0].legend(loc="upper left", fontsize=6.5)
    fig.suptitle("Original protocol (raw inputs, 20 epochs, SGD lr 0.01 vs Adam lr 0.001; 5 seeds)", fontsize=8,
                 x=0.01, ha="left", y=1.04, color=INK)
    fig.tight_layout(w_pad=0.8)
    save(fig, name)


# ---------------------------------------------------------------- Figure 5: generalization
def fig_generalization(sweeps=("main_cifar10_mlp", "main_mnist_mlp"), target=0.01, name="fig5_generalization"):
    sel = [("Original metric", "orig_rel_sharp_test"), ("Isotropic avg.", "avg_iso_0.01"),
           ("Hessian trace", "hess_trace"), ("λ_max", "lambda_max"), ("Minimum sharpness", "orbitmin_trace"),
           ("Orbit-min Σ√G", "orbitmin_sqrt_trace"), ("Σ w² G_ii", "ggn_trace_wscaled")]
    dfs = [(s, sweep_df(s)) for s in sweeps]
    dfs = [(s, d) for s, d in dfs if d is not None and d["opt"].nunique() >= 4]
    if not dfs:
        return
    fig, axs = plt.subplots(1, len(dfs), figsize=(3.5 * len(dfs), 2.3), squeeze=False)
    for ax, (s, d) in zip(axs[0], dfs):
        sub = d[d["target"] == target]
        ys = np.arange(len(sel))[::-1]
        for j, (disp, col) in enumerate(sel):
            pooled = kendalltau(sub[col], sub["gap_acc"])[0]
            within = np.mean([kendalltau(g[col], g["gap_acc"])[0] for _, g in sub.groupby("opt")])
            ax.plot([within, pooled], [ys[j]] * 2, color="#c3c2b7", lw=1)
            ax.plot(within, ys[j], marker="o", color="#86b6ef", ms=5, markeredgecolor="white",
                    label="within optimizer" if j == 0 else None)
            ax.plot(pooled, ys[j], marker="s", color="#1c5cab", ms=5, markeredgecolor="white",
                    label="across optimizers (pooled)" if j == 0 else None)
        ax.set_yticks(ys)
        ax.set_yticklabels([x[0] for x in sel], color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.1, 0.9)
        ax.axvline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel(f"Kendall τ with generalization gap  (train loss {target})")
        ax.set_title(DS_LABEL.get(s, s), loc="left")
    axs[0][0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- Figure 7: long-horizon label-noise dynamics
def load_orbit_dynamics():
    runs = {}
    for p in glob.glob(os.path.join(RES, "orbit_dynamics", "*.json")):
        r = json.load(open(p))
        c = r["cfg"]
        runs.setdefault((c["dataset"], c["opt"], c["lr"], c["label_noise"]), []).append(r["log"])
    return runs


def fig_labelnoise(name="fig7_labelnoise", lrs={"sgd": 0.12, "adam": 6e-4}):
    runs = load_orbit_dynamics()
    if not runs:
        return
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.35))
    panels = [("mnist", "excess1", "MNIST: trace (p = 1)"), ("mnist", "excess_half", "MNIST: Σ√G (p = ½)"),
              ("cifar10", "excess1", "CIFAR-10: trace (p = 1)"), ("cifar10", "excess_half", "CIFAR-10: Σ√G (p = ½)")]
    for ax, (ds, field, title) in zip(axs, panels):
        for opt in ["sgd", "adam"]:
            lab, col, mk, _ = STYLE[opt]
            for ln, ls, alpha in [(0.2, "-", 1.0), (0.0, ":", 0.9)]:
                logs = runs.get((ds, opt, lrs[opt], ln))
                if not logs:
                    continue
                n = min(len(lg) for lg in logs)
                ep = np.array([logs[0][i]["epoch"] for i in range(n)])
                v = np.array([[lg[i][field] for i in range(n)] for lg in logs])
                keep = ep >= 1
                ax.plot(ep[keep], v.mean(0)[keep], color=col, ls=ls, lw=1.5, alpha=alpha,
                        label=f"{lab}{' + label noise' if ln else ''}")
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_xlabel("epoch")
        ax.axhline(1.0, color="#c3c2b7", lw=0.8)
    axs[0].set_ylabel("orbit excess (raw / orbit minimum)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.8, handlelength=2.2,
               bbox_to_anchor=(0.5, -0.01))
    save(fig, name)


# ---------------------------------------------------------------- Figure 6: CNN
def fig_cnn(name="fig6_cnn"):
    d = sweep_df("cnn_cifar10")
    if d is None:
        return
    d = d.assign(excess=d["ggn_trace"] / d["orbitmin_trace"], l_raw=np.log2(d["ggn_trace"]),
                 l_fun=np.log2(d["orbitmin_trace"]))
    d = d.assign(l_coord=d["l_raw"] - d["l_fun"])
    targets = sorted(d["target"].unique(), reverse=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for opt in ["sgdm", "sgd", "adam"]:
        g = d[d["opt"] == opt].groupby("target")["excess"]
        med, lo_, hi = g.median(), g.min(), g.max()
        lab, c, mk, ls = STYLE[opt]
        ax.fill_between(med.index, lo_, hi, color=c, alpha=0.12, lw=0)
        ax.plot(med.index, med.values, color=c, marker=mk, ls=ls, label=lab, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.axhline(1.0, color="#c3c2b7", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(1.4, 0.007)
    ax.set_ylim(0.9, 3.9)
    ax.set_xlabel("training loss")
    ax.set_ylabel("orbit excess  tr G / min. sharpness")
    ax.set_title("(a) Orbit excess (median, min–max)", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    width = 0.36
    ticks, ticklabels, offset = [], [], 0
    for k, other in enumerate(["sgd", "sgdm"]):
        xs = np.arange(len(targets)) + offset
        offset += len(targets) + 1
        fun, coord, raw = [], [], []
        for t in targets:
            a, x = d[(d.opt == "adam") & (d.target == t)], d[(d.opt == other) & (d.target == t)]
            fun.append(a.l_fun.mean() - x.l_fun.mean())
            coord.append(a.l_coord.mean() - x.l_coord.mean())
            raw.append(a.l_raw.mean() - x.l_raw.mean())
        bx.bar(xs - width / 2, fun, width=width, color="#1c5cab", label="function (minimum sharpness)" if k == 0 else None)
        bx.bar(xs + width / 2, coord, width=width, color="#86b6ef", label="coordinates (orbit excess)" if k == 0 else None)
        bx.plot(xs, raw, ls="none", marker="D", ms=5, color=INK, markeredgecolor="white", markeredgewidth=0.6,
                label="raw trace = sum" if k == 0 else None)
        bx.text(xs.mean(), -1.25, f"Adam vs {STYLE[other][0]}", ha="center", fontsize=6.8, color=INK2)
        ticks += list(xs)
        ticklabels += [f"{t:g}".replace("0.", ".") for t in targets]
    bx.axhline(0, color="#c3c2b7", lw=0.8)
    bx.set_xticks(ticks)
    bx.set_xticklabels(ticklabels, fontsize=6)
    bx.set_xlabel("training loss at which the comparison is made", fontsize=7)
    bx.set_ylim(-1.4, 2.6)
    bx.set_ylabel("log₂(Adam / X) of Hessian trace\n← Adam flatter      Adam sharper →", fontsize=7)
    bx.legend(loc="upper right", fontsize=6.2, handlelength=1.2)
    bx.grid(axis="x", visible=False)
    bx.set_title("(b) Raw trace = function × coordinates", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


if __name__ == "__main__":
    fig_flip(["main_cifar10_mlp", "main_mnist_mlp"])
    fig_cnn()
    fig_mechanism()
    fig_intervention()
    fig_replication()
    fig_generalization()
