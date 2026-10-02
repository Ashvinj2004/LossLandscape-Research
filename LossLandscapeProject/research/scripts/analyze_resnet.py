"""Analysis of the ResNet-18 / full CIFAR-10 sweeps produced by scripts/resnet_cifar10.py.

For each sweep and loss target:
  * medians of every measure per configuration (optimizer, learning rate, weight decay);
  * P(A rated flatter than B) over seed-matched configuration pairs, for raw, BN-invariant and
    per-weight-invariant measures (as in the paper's audit);
  * the exact split log2(trH_A / trH_B) = log2(ntrace_A / ntrace_B) + log2(norm factor_A / norm factor_B),
    where ntrace is the trace at unit-norm conv filters (invariant to the BN scale symmetry) and the norm
    factor trH / ntrace is set by the filter norms;
  * per learning-rate pairing win counts, and Spearman correlations with the conv filter norms.

usage: python scripts/analyze_resnet.py [--results DIR] [--json out.json]
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

MEASURES = {  # name -> (column, family)
    "orig. rel. iso (test)": ("orig_rel_sharp_test", "raw"),
    "rel. iso test, s=1e-3": ("rel_iso_test_0.001", "raw"),
    "iso avg, s=1e-3": ("avg_iso_0.001", "raw"),
    "tr H": ("hess_trace", "raw"),
    "lambda_max": ("lambda_max", "raw"),
    "SAM worst, r=0.05": ("worst_sam_0.05", "raw"),
    "ntrace (unit filters)": ("ntrace", "bn-inv"),
    "filter avg, s=0.03": ("avg_filter_0.03", "bn-inv"),
    "sum w^2 H_ii": ("wtrace", "inv"),
    "mult avg, s=0.03": ("avg_mult_0.03", "inv"),
    "ASAM worst, r=0.5": ("worst_asam_0.5", "inv"),
}


def load(results, sweep):
    rows = []
    for p in sorted(glob.glob(os.path.join(results, sweep, "*.json"))):
        if p.endswith(".partial.json"):
            continue
        r = json.load(open(p))
        c = r["cfg"]
        for ck in r["ckpts"]:
            row = {k: c[k] for k in ("opt", "lr", "wd", "seed")}
            row.update({"run": os.path.basename(p)[:-5], "status": r["status"],
                        "target": ck["target"] if ck["target"] is not None else "final"})
            row.update({k: v for k, v in ck.items() if not isinstance(v, (list, dict)) and k != "target"})
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["config"] = df["opt"] + "_lr" + df["lr"].map(lambda v: f"{v:g}") + np.where(df["wd"] > 0, "_wd" + df["wd"].map(lambda v: f"{v:g}"), "")
        df["norm_factor"] = df["hess_trace"] / df["ntrace"]
    return df


def P(df, col, sel_a, sel_b):
    w = n = 0.0
    for seed in df["seed"].unique():
        xa = df[sel_a(df) & (df.seed == seed)][col].dropna().values
        xb = df[sel_b(df) & (df.seed == seed)][col].dropna().values
        for u, v in itertools.product(xa, xb):
            w += 1.0 if u < v else 0.5 if u == v else 0.0
            n += 1
    return (w / n if n else np.nan), int(n)


def log2_ratio(df, col, sel_a, sel_b):
    a, b = df[sel_a(df)][col].dropna(), df[sel_b(df)][col].dropna()
    if len(a) == 0 or len(b) == 0 or (a <= 0).any() or (b <= 0).any():
        return np.nan
    return float(np.log2(a).mean() - np.log2(b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for sweep in ["resnet18_noaug", "resnet18_aug"]:
        df = load(a.results, sweep)
        if df.empty:
            continue
        print(f"\n######## {sweep}: {df['run'].nunique()} runs, {len(df)} checkpoints ########")
        res = out[sweep] = {}
        print("\nRuns per configuration and target:")
        print(df.pivot_table(index="config", columns="target", values="run", aggfunc="nunique"))
        cols = ["epoch", "train_loss", "test_acc", "conv_filter_norm_gmean", "hess_trace", "ntrace", "norm_factor",
                "wtrace", "lambda_max", "lambda_max_conv", "avg_iso_0.001", "avg_filter_0.03", "avg_mult_0.03",
                "worst_sam_0.05", "worst_asam_0.5", "orig_rel_sharp_test", "rel_iso_test_0.001"]
        res["median"] = {}
        for col in [c for c in cols if c in df]:
            t = df.pivot_table(index="config", columns="target", values=col, aggfunc="median")
            print(f"\nmedian {col}:")
            print(t.round(4))
            res["median"][col] = json.loads(t.to_json())
        opts = sorted(df["opt"].unique())
        adam_like = [o for o in opts if o.startswith("adam")]
        sgd_like = [o for o in opts if o.startswith("sgd")]
        res["P"], res["split"] = {}, {}
        for A, B in itertools.product(adam_like, sgd_like):
            sa, sb = (lambda d, o=A: d.opt == o), (lambda d, o=B: d.opt == o)
            key = f"{A}_vs_{B}"
            print(f"\n===== P({A} rated flatter than {B}) / log2({A}/{B}) =====")
            rows = []
            for name, (col, fam) in MEASURES.items():
                if col not in df:
                    continue
                row = [name, fam]
                for t in [1.0, 0.1, 0.01, "final"]:
                    d = df[df.target == t]
                    if len(d) == 0:
                        continue
                    p, npairs = P(d, col, sa, sb)
                    row += [p, log2_ratio(d, col, sa, sb)]
                    res["P"].setdefault(key, {}).setdefault(col, {})[str(t)] = p
                rows.append(row)
            tg = [t for t in [1.0, 0.1, 0.01, "final"] if (df.target == t).any()]
            print(pd.DataFrame(rows, columns=["measure", "type"] + [f"{k}@{t}" for t in tg for k in ("P", "log2")]).round(2).to_string(index=False))
            print("exact split of the trace ratio (bits): raw = BN-invariant (ntrace) + filter norms")
            for t in tg:
                d = df[df.target == t]
                raw, inv, nf = (log2_ratio(d, c, sa, sb) for c in ("hess_trace", "ntrace", "norm_factor"))
                res["split"].setdefault(key, {})[str(t)] = {"raw": raw, "ntrace": inv, "norm": nf}
                print(f"  target {t}: raw {raw:+.2f} = ntrace {inv:+.2f} + norms {nf:+.2f}")
        print("\nSpearman correlation with the conv filter norm (geometric mean of per-layer medians):")
        res["spearman_norm"] = {}
        for col in ["hess_trace", "lambda_max", "avg_iso_0.001", "ntrace", "wtrace", "worst_asam_0.5", "avg_mult_0.03"]:
            if col not in df:
                continue
            rs = {str(t): float(df[df.target == t][[col, "conv_filter_norm_gmean"]].corr("spearman").iloc[0, 1])
                  for t in [1.0, 0.1, 0.01, "final"] if (df.target == t).sum() > 2}
            res["spearman_norm"][col] = rs
            print(f"  {col:16s}", "  ".join(f"{t}: {r:+.2f}" for t, r in rs.items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
