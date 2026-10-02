"""Analysis of the depth sweep (results/depth_cifar10): does the coordinate factor grow with depth?

For each depth (hidden layers of width 128, He init) and loss target: median orbit excess per optimizer,
the exact split of log2(tr G_Adam / tr G_X) (geometric means over runs) into a function term
(minimum sharpness) and a coordinate term (orbit excess), and P(Adam rated flatter than X) under the raw
trace and under minimum sharpness (pairs share a seed).

usage: python scripts/analyze_depth.py [--json out.json]
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

OPTS = ["sgd", "sgdm", "adam"]


def P(df, col, a, b):
    w = n = 0.0
    for seed in df["seed"].unique():
        xa = df[(df.opt == a) & (df.seed == seed)][col].values
        xb = df[(df.opt == b) & (df.seed == seed)][col].values
        for u, v in itertools.product(xa, xb):
            w += 1.0 if u < v else 0.5 if u == v else 0.0
            n += 1
    return w / n if n else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    df = load_sweep("depth_cifar10")
    df = df[df["target"].notna()].copy()
    df["depth"] = df["arch"].str.extract(r"mlp_d(\d+)_").astype(int)
    df["excess"] = df["ggn_trace"] / df["orbitmin_trace"]
    print(f"{df['run'].nunique()} runs, {len(df)} loss-matched checkpoints")
    print("\nRuns reaching each target (depth x opt):")
    print(df.pivot_table(index=["depth", "opt"], columns="target", values="run", aggfunc="nunique"))
    out = {"excess": {}, "split": {}, "P": {}}
    print("\nMedian orbit excess:")
    t = df.pivot_table(index=["depth", "opt"], columns="target", values="excess", aggfunc="median")
    print(t.round(2))
    out["excess"] = {f"d{d}_{o}": {str(k): float(v) for k, v in t.loc[(d, o)].items()} for d, o in t.index}
    print("\nMedian test accuracy:")
    print(df.pivot_table(index=["depth", "opt"], columns="target", values="test_acc", aggfunc="median").round(3))
    for other in ["sgd", "sgdm"]:
        print(f"\n===== Adam vs {other}: log2 split (raw = function + coordinates), P(flatter) raw / MS1 =====")
        rows = []
        for (d, tgt), g in df.groupby(["depth", "target"]):
            A, B = g[g.opt == "adam"], g[g.opt == other]
            if len(A) == 0 or len(B) == 0:
                continue
            fun = np.log2(A.orbitmin_trace).mean() - np.log2(B.orbitmin_trace).mean()
            coord = np.log2(A.excess).mean() - np.log2(B.excess).mean()
            rows.append(dict(depth=d, target=tgt, n=f"{len(A)}x{len(B)}", raw=fun + coord, function=fun,
                             coordinates=coord, P_raw=P(g, "ggn_trace", "adam", other),
                             P_ms1=P(g, "orbitmin_trace", "adam", other),
                             P_iso=P(g, "avg_iso_0.01", "adam", other), P_mult=P(g, "avg_mult_0.03", "adam", other)))
        r = pd.DataFrame(rows)
        print(r.round(2).to_string(index=False))
        out["split"][other] = r.to_dict(orient="records")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
