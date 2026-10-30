"""Label-noise destinations (Sec. 5.5) tested at the level of independent initializations.

Per run: the change in log orbit excess from epoch 100 to the end at p = 1/2 minus that at p = 1
(> 0: the trace excess fell by more, as predicted for SGD; < 0: the sum-sqrt(G) excess fell by more, as
predicted for Adam). Runs that share a dataset and a seed start from the same network (the learning
rates and the orbit starts c of coordinate_memory are variations on one initialization), so they form
one block; the block statistic is the mean over its runs, and the sign test counts blocks. Blocks merge
the original and the mirrored CIFAR-10 data for seeds 0 and 1, which is conservative.

Frozen-curvature test (label_noise_seeds only): per run, the summed change of the frozen objectives
under the orbit component of the motion between snapshots (epochs >= 100), p = 1 minus p = 1/2
(< 0: the orbit motion lowers the trace objective by more, as predicted for SGD).

usage: python scripts/analyze_label_noise.py [--json results/label_noise_summary.json]
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
EXPS = ["orbit_dynamics", "coordinate_memory", "label_noise_seeds"]


def excess_stat(log):
    e = [s for s in log if s["epoch"] >= 100]
    a, b = e[0], e[-1]
    return (np.log(b["excess_half"]) - np.log(a["excess_half"])) - (np.log(b["excess1"]) - np.log(a["excess1"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    runs = collections.defaultdict(list)
    frozen = collections.defaultdict(list)
    for exp in EXPS:
        for f in glob.glob(os.path.join(RES, exp, "*.json")):
            r = json.load(open(f))
            c = r["cfg"]
            if c.get("label_noise", 0) == 0:
                continue
            runs[c["opt"]].append({"exp": exp, "dataset": c["dataset"], "seed": c["seed"], "stat": excess_stat(r["log"])})
            if r.get("frozen"):
                d1 = sum(x["dlog_frozen1"] for x in r["frozen"])
                dh = sum(x["dlog_frozen_half"] for x in r["frozen"])
                frozen[c["opt"]].append({"dataset": c["dataset"], "seed": c["seed"], "d1": d1, "dhalf": dh, "diff": d1 - dh,
                                         "max_q_residual": max(x["q_residual"] for x in r["frozen"])})
    out = {}
    for opt, rows in sorted(runs.items()):
        sign = 1 if opt == "sgd" else -1  # predicted sign of the statistic
        blocks = collections.defaultdict(list)
        for x in rows:
            blocks[(x["dataset"], x["seed"])].append(x["stat"])
        bstat = {k: float(np.mean(v)) for k, v in blocks.items()}
        k_ok = sum(sign * v > 0 for v in bstat.values())
        n_runs_ok = sum(sign * x["stat"] > 0 for x in rows)
        res = {"runs": len(rows), "runs_in_predicted_direction": int(n_runs_ok), "blocks": len(bstat),
               "blocks_in_predicted_direction": int(k_ok),
               "sign_test_p_blocks": float(binomtest(k_ok, len(bstat), 0.5).pvalue),
               "mean_block_stat": float(np.mean(list(bstat.values()))),
               "blocks_per_dataset": {ds: sum(1 for k in bstat if k[0] == ds) for ds in ("mnist", "cifar10")}}
        fz = frozen.get(opt, [])
        if fz:
            fsign = -1 if opt == "sgd" else 1  # predicted sign of d1 - dhalf
            kf = sum(fsign * x["diff"] > 0 for x in fz)
            res["frozen"] = {"runs": len(fz), "in_predicted_direction": int(kf),
                             "sign_test_p": float(binomtest(kf, len(fz), 0.5).pvalue),
                             "mean_d1": float(np.mean([x["d1"] for x in fz])),
                             "mean_dhalf": float(np.mean([x["dhalf"] for x in fz])),
                             "mean_diff": float(np.mean([x["diff"] for x in fz])),
                             "max_q_residual": float(max(x["max_q_residual"] for x in fz)),
                             "per_dataset": {ds: [round(x["diff"], 4) for x in fz if x["dataset"] == ds] for ds in ("mnist", "cifar10")}}
        out[opt] = res
        print(f"== {opt}: {res['runs']} runs ({res['runs_in_predicted_direction']} in the predicted direction), "
              f"{res['blocks']} independent blocks ({res['blocks_in_predicted_direction']} in the predicted direction), "
              f"sign test p = {res['sign_test_p_blocks']:.2g}; mean block statistic {res['mean_block_stat']:+.3f}")
        if fz:
            q = res["frozen"]
            print(f"   frozen curvature: {q['in_predicted_direction']}/{q['runs']} runs in the predicted direction, "
                  f"p = {q['sign_test_p']:.2g}; mean dlog frozen p=1 {q['mean_d1']:+.4f}, p=1/2 {q['mean_dhalf']:+.4f}; "
                  f"max Q residual {q['max_q_residual']:.1e}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
