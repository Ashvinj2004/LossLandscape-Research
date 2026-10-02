"""Analysis of the coordinate-memory experiment (results/coordinate_memory).

(a) Memory of the starting point. The same initial function is started at three orbit points
    (c in {1/3, 1, 3}). For each optimizer/noise setting we report the orbit excess (p=1 and p=1/2) per
    c over training, its spread across c (max/min ratio), and how far the conserved quantities Q_j
    move: rel_dQ = ||Q(t) - Q(0)|| / ||Q(0)|| per layer, and the across-c separation of the mean Q.
(b) Drift direction. Between consecutive snapshots, each hidden unit's change in Q_j is correlated
    (Spearman, across units) with its curvature imbalance log(A_j/B_j) at the earlier snapshot, at
    p=1 (trace balance) and p=1/2 (sum-sqrt balance). Drift toward balance at power p predicts a
    positive correlation at that p. Reported for the early window (epochs <= 25) and the late window
    (epochs >= 100), averaged over snapshot pairs, units of both layers pooled per pair.

usage: python scripts/analyze_coordinate_memory.py [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "coordinate_memory")
CS = [1 / 3, 1.0, 3.0]
SETTINGS = [("sgd", 0.0), ("sgd", 0.2), ("adam", 0.0), ("adam", 0.2)]
LABEL = {("sgd", 0.0): "SGD", ("sgd", 0.2): "SGD + label noise", ("adam", 0.0): "Adam", ("adam", 0.2): "Adam + label noise"}
REPORT_EPOCHS = [0, 1, 3, 5, 12, 25, 50, 100, 200, 400, 600, 800]


def load():
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        runs.append(r)
    return runs


def at(log, epoch, key):
    for s in log:
        if s["epoch"] == epoch:
            return s[key]
    return None


def drift_corr(log, lo, hi, key):
    """Mean over consecutive snapshot pairs (both epochs in [lo, hi]) of Spearman(dQ_j, logAB_j)."""
    rs = []
    for s0, s1 in zip(log[:-1], log[1:]):
        if s0["epoch"] < lo or s1["epoch"] > hi:
            continue
        dq = np.concatenate([np.array(q1) - np.array(q0) for q0, q1 in zip(s0["Q"], s1["Q"])])
        lab = np.concatenate([np.array(x) for x in s0[key]])
        if np.std(dq) == 0:
            continue
        rs.append(spearmanr(dq, lab).correlation)
    return float(np.mean(rs)) if rs else np.nan


def rms_dq(log, epoch, layer):
    """(RMS of Q_j(epoch) - Q_j(0), RMS of Q_j(0)) over the units of a layer."""
    q0, q = np.array(log[0]["Q"][layer]), at(log, epoch, "Q")
    if q is None:
        return np.nan, np.nan
    return float(np.sqrt(np.mean((np.array(q[layer]) - q0) ** 2))), float(np.sqrt(np.mean(q0 ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = load()
    print(f"{len(runs)} runs")
    out = {"n_runs": len(runs), "settings": {}}
    for ds in ["mnist", "cifar10"]:
        for opt, ln in SETTINGS:
            sel = [r for r in runs if r["cfg"]["dataset"] == ds and r["cfg"]["opt"] == opt and r["cfg"]["label_noise"] == ln]
            if not sel:
                continue
            key = f"{ds}_{opt}_ln{ln}"
            res = {"n": len(sel), "excess1": {}, "excess_half": {}, "spread1": {}, "spread_half": {},
                   "rel_dQ0": {}, "rel_dQ1": {}, "meanQ0": {}, "train_loss": {}, "test_acc": {}}
            print(f"\n================ {ds}  {LABEL[(opt, ln)]}  ({len(sel)} runs) ================")
            epochs = [e for e in REPORT_EPOCHS if e <= sel[0]["cfg"]["epochs"]]
            for metric in ["excess1", "excess_half"]:
                print(f"{metric} (mean over seeds) per c:")
                print("   epoch " + " ".join(f"{e:>6d}" for e in epochs))
                per_c = {}
                for c in CS:
                    rr = [r for r in sel if np.isclose(r["cfg"]["c"], c)]
                    vals = [np.nanmean([at(r["log"], e, metric) if at(r["log"], e, metric) is not None else np.nan for r in rr])
                            if rr else np.nan for e in epochs]
                    per_c[c] = vals
                    res[metric][f"{c:.3g}"] = dict(zip(map(str, epochs), vals))
                    print(f"  c={c:<5.3g} " + " ".join(f"{v:6.2f}" for v in vals))
                spread = [np.nanmax([per_c[c][i] for c in CS]) / np.nanmin([per_c[c][i] for c in CS]) for i in range(len(epochs))]
                res["spread1" if metric == "excess1" else "spread_half"] = dict(zip(map(str, epochs), spread))
                print("  max/min " + " ".join(f"{v:6.2f}" for v in spread))
            for c in CS:
                rr = [r for r in sel if np.isclose(r["cfg"]["c"], c)]
                if not rr:
                    continue
                last = rr[0]["log"][-1]["epoch"]
                res["rel_dQ0"][f"{c:.3g}"] = np.mean([rms_dq(r["log"], last, 0) for r in rr], axis=0).tolist()
                res["rel_dQ1"][f"{c:.3g}"] = np.mean([rms_dq(r["log"], last, 1) for r in rr], axis=0).tolist()
                res["meanQ0"][f"{c:.3g}"] = {str(e): float(np.mean([np.mean(at(r["log"], e, "Q")[0]) for r in rr if at(r["log"], e, "Q")]))
                                             for e in [0, last]}
                res["train_loss"][f"{c:.3g}"] = float(np.mean([r["log"][-1]["train_loss"] for r in rr]))
                res["test_acc"][f"{c:.3g}"] = float(np.mean([r["log"][-1]["test_acc"] for r in rr]))
            print("final: RMS(Q-Q0) [RMS Q0] layer0 / layer1, mean Q layer0 (start -> end), train loss, test acc per c:")
            for c in CS:
                k = f"{c:.3g}"
                if k in res["rel_dQ0"]:
                    mq = list(res["meanQ0"][k].values())
                    d0, d1 = res["rel_dQ0"][k], res["rel_dQ1"][k]
                    print(f"  c={k:<5s} {d0[0]:.3f} [{d0[1]:.3f}] / {d1[0]:.3f} [{d1[1]:.3f}]   meanQ0 {mq[0]:+.3f} -> {mq[-1]:+.3f}"
                          f"   loss {res['train_loss'][k]:.3f}  acc {res['test_acc'][k]:.3f}")
            res["drift_corr"], res["drift_diff"] = {}, {}
            for win, (lo, hi) in {"early": (0, 25), "late": (100, 10 ** 6)}.items():
                for p, kk in [("1", "logAB1"), ("half", "logABhalf")]:
                    vals = [drift_corr(r["log"], lo, hi, kk) for r in sel]
                    res["drift_corr"][f"{win}_p{p}"] = {"mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)),
                                                        "max": float(np.nanmax(vals)), "n_pos": int(np.sum(np.array(vals) > 0)),
                                                        "n": int(np.sum(~np.isnan(vals)))}
                d = [drift_corr(r["log"], lo, hi, "logAB1") - drift_corr(r["log"], lo, hi, "logABhalf") for r in sel]
                res["drift_diff"][win] = {"mean": float(np.nanmean(d)), "n_pos": int(np.sum(np.array(d) > 0)), "n": len(d)}
            print("Spearman(dQ_j, log A_j/B_j), mean [min, max] over runs, (#positive/#runs):")
            for k, v in res["drift_corr"].items():
                print(f"  {k:12s} {v['mean']:+.3f} [{v['min']:+.3f}, {v['max']:+.3f}]  ({v['n_pos']}/{v['n']})")
            for win, v in res["drift_diff"].items():
                print(f"  {win}: corr(p=1) - corr(p=1/2) = {v['mean']:+.3f}, positive in {v['n_pos']}/{v['n']} runs")
            out["settings"][key] = res
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
