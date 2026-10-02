"""Analysis of the main sweeps (loss-matched checkpoints).

usage: python scripts/analyze_main.py main_cifar10_mlp [main_mnist_mlp ...]
Prints the key tables; figures are produced by make_figures.py.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls.analysis import load_sweep  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

# measure name -> (column, family). "raw" depends on the parameterization, "canon" is evaluated at the
# balanced (min-norm) point of the rescaling orbit, "inv" is rescaling-invariant.
MEASURES = {
    "orig. rel. iso (test)": ("orig_rel_sharp_test", "raw"),
    "iso avg, s=0.01": ("avg_iso_0.01", "raw"),
    "tr H (Hutchinson)": ("hess_trace", "raw"),
    "tr G": ("ggn_trace", "raw"),
    "lambda_max": ("lambda_max", "raw"),
    "SAM worst, r=0.05": ("worst_sam_0.05", "raw"),
    "sum sqrt(G_ii)": ("ggn_sqrt_trace", "raw"),
    "bal: iso avg, s=0.01": ("bal_avg_iso_0.01", "canon"),
    "bal: tr G": ("bal_ggn_trace", "canon"),
    "bal: lambda_max": ("bal_lambda_max", "canon"),
    "bal: SAM worst": ("bal_worst_sam_0.05", "canon"),
    "orbit-min tr G": ("orbitmin_trace", "inv"),
    "orbit-min sum sqrt(G_ii)": ("orbitmin_sqrt_trace", "inv"),
    "mult avg, s=0.03": ("avg_mult_0.03", "inv"),
    "filter avg, s=0.03": ("avg_filter_0.03", "inv"),
    "ASAM worst, r=0.5": ("worst_asam_0.5", "inv"),
    "sum w^2 G_ii": ("ggn_trace_wscaled", "inv"),
}
OPTS = ["sgd", "sgdm", "sam", "adam"]


def prob_flatter(df, col, a, b, target):
    """P(measure(A) < measure(B)) over all (config_A, config_B) pairs at the same loss target, same seed."""
    sub = df[df["target"] == target]
    wins, n = 0.0, 0
    for seed in sub["seed"].unique():
        xa = sub[(sub["opt"] == a) & (sub["seed"] == seed)][col].dropna().values
        xb = sub[(sub["opt"] == b) & (sub["seed"] == seed)][col].dropna().values
        for u, v in itertools.product(xa, xb):
            wins += 1.0 if u < v else (0.5 if u == v else 0.0)
            n += 1
    return wins / n if n else np.nan, n


def main(names):
    df = pd.concat([load_sweep(n).assign(sweep=n) for n in names], ignore_index=True)
    df = df[df["target"].notna()]
    print(f"{len(df)} loss-matched checkpoints from {df['run'].nunique()} runs")
    for sweep, d in df.groupby("sweep"):
        print(f"\n==================== {sweep} ====================")
        print("\nRuns reaching each target, per optimizer:")
        print(d.pivot_table(index="opt", columns="target", values="run", aggfunc="nunique").reindex(OPTS))
        print("\nImbalance (mean |log ||in||/||out|| |), layer 0 / layer 1, median over runs:")
        print(d.pivot_table(index="opt", columns="target", values="imbalance_0", aggfunc="median").reindex(OPTS).round(3))
        print(d.pivot_table(index="opt", columns="target", values="imbalance_1", aggfunc="median").reindex(OPTS).round(3))
        print("\nRescaling inflation kappa = bal tr G / raw tr G (median):")
        d = d.assign(kappa=d["bal_ggn_trace"] / d["ggn_trace"], kappa_iso=d["bal_avg_iso_0.01"] / d["avg_iso_0.01"],
                     kappa_lam=d["bal_lambda_max"] / d["lambda_max"])
        print(d.pivot_table(index="opt", columns="target", values="kappa", aggfunc="median").reindex(OPTS).round(2))
        d = d.assign(excess_tr=d["ggn_trace"] / d["orbitmin_trace"], excess_sqrt=d["ggn_sqrt_trace"] / d["orbitmin_sqrt_trace"])
        print("Excess of raw tr G over its orbit minimum (median):")
        print(d.pivot_table(index="opt", columns="target", values="excess_tr", aggfunc="median").reindex(OPTS).round(3))
        print("Excess of raw sum sqrt(G_ii) over its orbit minimum (median):")
        print(d.pivot_table(index="opt", columns="target", values="excess_sqrt", aggfunc="median").reindex(OPTS).round(3))
        print("kappa for iso sharpness:")
        print(d.pivot_table(index="opt", columns="target", values="kappa_iso", aggfunc="median").reindex(OPTS).round(2))
        print("kappa for lambda_max:")
        print(d.pivot_table(index="opt", columns="target", values="kappa_lam", aggfunc="median").reindex(OPTS).round(2))
        for tgt in [0.3, 0.1, 0.01]:
            print(f"\nP(Adam flatter than X) at train loss {tgt}  [n pairs]:")
            rows = []
            for name, (col, fam) in MEASURES.items():
                if col not in d:
                    continue
                row = {"measure": name, "family": fam}
                for b in ["sgd", "sgdm", "sam"]:
                    p, n = prob_flatter(d, col, "adam", b, tgt)
                    row[f"vs {b}"] = round(p, 2)
                    row.setdefault("n", n)
                rows.append(row)
            print(pd.DataFrame(rows).to_string(index=False))
        print("\nMedian measure per optimizer at train loss 0.1:")
        sub = d[d["target"] == 0.1]
        print(sub.groupby("opt")[[c for c, _ in MEASURES.values() if c in sub]].median().reindex(OPTS).T.round(4))
        print("\nTest accuracy / loss gap at train loss 0.01 (median):")
        sub = d[d["target"] == 0.01]
        print(sub.groupby("opt")[["test_acc", "gap_loss"]].median().reindex(OPTS).round(4))


if __name__ == "__main__":
    main(sys.argv[1:])
