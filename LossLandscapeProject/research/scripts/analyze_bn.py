"""Analysis of the BatchNorm sweep (results/bn_cifar10).

BatchNorm adds a scale symmetry per pre-BN row (the loss is invariant to w_row -> s w_row), on top of
the ReLU rescaling between the BN affine parameters and the next layer. Raw measures (trace, lambda_max,
isotropic, SAM, the original metric) depend on both; the per-weight-invariant measures (sum w^2 H_ii,
multiplicative average, ASAM) do not. Filter-normalized sharpness is invariant to the BN row scaling
only (scripts/test_invariance.py). There is no orbit-minimum machinery here.

usage: python scripts/analyze_bn.py [--json out.json]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls.analysis import load_sweep  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

MEASURES = {
    "orig. rel. iso (test)": ("orig_rel_sharp_test", "raw"),
    "iso avg, s=0.01": ("avg_iso_0.01", "raw"),
    "tr H (Hutchinson)": ("hess_trace", "raw"),
    "lambda_max": ("lambda_max", "raw"),
    "SAM worst, r=0.05": ("worst_sam_0.05", "raw"),
    "sum w^2 H_ii": ("wtrace", "inv"),
    "mult avg, s=0.03": ("avg_mult_0.03", "inv"),
    "filter avg, s=0.03": ("avg_filter_0.03", "row"),
    "ASAM worst, r=0.5": ("worst_asam_0.5", "inv"),
}
GROUPS = ["sgd", "sgdm", "adam", "sgdm+wd", "adamw"]
# "group" or "group@lr". Weight-decay effects are compared at the same learning rate.
COMPARISONS = [("adam", "sgd"), ("adam", "sgdm"), ("adamw", "sgdm+wd"), ("adamw", "adam@0.001"),
               ("sgdm+wd", "sgdm@0.03")]
TARGETS = [1.0, 0.3, 0.1]


def select(df, spec):
    g, _, lr = spec.partition("@")
    m = df["group"] == g
    return m & np.isclose(df["lr"], float(lr)) if lr else m


def load():
    df = load_sweep("bn_cifar10")
    wd = df["wd"].fillna(0.0) if "wd" in df else pd.Series(0.0, index=df.index)
    df["group"] = np.where((df["opt"] == "sgdm") & (wd > 0), "sgdm+wd", df["opt"])
    return df[df["target"].notna()].copy()


def prob_flatter(df, col, a, b, target):
    """P(measure(A) < measure(B)) over all (config_A, config_B) pairs at the same loss target and seed."""
    sub = df[df["target"] == target]
    wins, n = 0.0, 0
    for seed in sub["seed"].unique():
        xa = sub[select(sub, a) & (sub["seed"] == seed)][col].dropna().values
        xb = sub[select(sub, b) & (sub["seed"] == seed)][col].dropna().values
        for u, v in itertools.product(xa, xb):
            wins += 1.0 if u < v else (0.5 if u == v else 0.0)
            n += 1
    return (wins / n if n else np.nan), n


def log2_ratio(df, col, a, b, target):
    """log2 of the ratio of geometric means (A over B) at a loss target."""
    sub = df[df["target"] == target]
    xa, xb = sub[select(sub, a)][col].dropna(), sub[select(sub, b)][col].dropna()
    if len(xa) == 0 or len(xb) == 0 or (xa <= 0).any() or (xb <= 0).any():
        return np.nan
    return float(np.log2(xa).mean() - np.log2(xb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    df = load()
    out = {"n_runs": int(df["run"].nunique()), "n_ckpts": int(len(df))}
    print(f"{len(df)} loss-matched checkpoints from {df['run'].nunique()} runs")
    print("\nRuns reaching each target:")
    reach = df.pivot_table(index="group", columns="target", values="run", aggfunc="nunique").reindex(GROUPS)
    print(reach)
    out["reach"] = {g: {str(t): (None if pd.isna(reach.loc[g, t]) else int(reach.loc[g, t])) for t in reach.columns} for g in GROUPS}

    stats = ["prebn_row_norm_0", "prebn_row_norm_1", "test_acc", "epoch", "train_loss"] + [c for c, _ in MEASURES.values()]
    out["median"] = {}
    for col in stats:
        t = df.pivot_table(index="group", columns="target", values=col, aggfunc="median").reindex(GROUPS)
        print(f"\nmedian {col}:")
        print(t.round(4))
        out["median"][col] = {g: {str(k): float(t.loc[g, k]) for k in t.columns} for g in GROUPS}

    out["P"], out["log2"] = {}, {}
    for A, B in COMPARISONS:
        key = f"{A}_vs_{B}"
        out["P"][key], out["log2"][key] = {}, {}
        print(f"\n===== P({A} rated flatter than {B}) and log2 geo-mean ratio ({A}/{B}) =====")
        rows = []
        for name, (col, fam) in MEASURES.items():
            ps = [prob_flatter(df, col, A, B, t) for t in TARGETS]
            lr = [log2_ratio(df, col, A, B, t) for t in TARGETS]
            out["P"][key][col] = {str(t): p for t, (p, _) in zip(TARGETS, ps)}
            out["log2"][key][col] = {str(t): v for t, v in zip(TARGETS, lr)}
            rows.append([name, fam] + [p for p, _ in ps] + lr + [ps[0][1]])
        cols = ["measure", "type"] + [f"P@{t}" for t in TARGETS] + [f"log2@{t}" for t in TARGETS] + ["n_pairs@1"]
        print(pd.DataFrame(rows, columns=cols).round(2).to_string(index=False))

    # Raw verdicts depend on the learning-rate pairing; count Adam wins per pairing over all targets.
    print("\nAdam rated flatter, per learning-rate pairing (wins/pairs over seeds and targets):")
    out["pairings"] = {}
    for col in ["hess_trace", "avg_iso_0.01", "lambda_max", "wtrace", "worst_asam_0.5", "avg_mult_0.03"]:
        row = {}
        for la, (gb, lb) in itertools.product([3e-4, 1e-3], [("sgd", 0.1), ("sgd", 0.3), ("sgdm", 0.01), ("sgdm", 0.03)]):
            w = n = 0
            for t, seed in itertools.product(TARGETS, range(3)):
                sub = df[(df["target"] == t) & (df["seed"] == seed)]
                xa = sub[select(sub, f"adam@{la}")][col].values
                xb = sub[select(sub, f"{gb}@{lb}")][col].values
                for u, v in itertools.product(xa, xb):
                    w, n = w + (u < v), n + 1
            row[f"adam{la:g}_vs_{gb}{lb:g}"] = [int(w), n]
        out["pairings"][col] = row
        print(f"  {col:16s}", "  ".join(f"{k}:{w}/{n}" for k, (w, n) in row.items()),
              f"  total {sum(w for w, _ in row.values())}/{sum(n for _, n in row.values())}")

    # Raw curvature w.r.t. a pre-BN row scales as 1/||row||^2; how much of the raw spread do the norms explain?
    print("\nSpearman correlation with the median first-layer pre-BN row norm, per target (all 24 runs):")
    out["spearman_norm0"] = {}
    for col in ["hess_trace", "lambda_max", "avg_iso_0.01", "wtrace", "worst_asam_0.5"]:
        rs = {str(t): float(df[df["target"] == t][[col, "prebn_row_norm_0"]].corr("spearman").iloc[0, 1]) for t in TARGETS}
        out["spearman_norm0"][col] = rs
        print(f"  {col:16s}", "  ".join(f"{t}: {r:+.2f}" for t, r in rs.items()))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
