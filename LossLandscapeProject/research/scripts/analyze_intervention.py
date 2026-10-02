"""Adam vs Adam-Q (drift removed) at matched training loss.

usage: python scripts/analyze_intervention.py main_cifar10_mlp interv_cifar10 [main_mnist_mlp interv_mnist]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lls.analysis import load_sweep  # noqa: E402
from analyze_main import prob_flatter  # noqa: E402

COLS = [("excess", "orbit excess"), ("imbalance_0", "imbalance L1"), ("ggn_trace", "tr G"),
        ("hess_trace", "tr H"), ("avg_iso_0.01", "iso avg"), ("orig_rel_sharp_test", "original"),
        ("lambda_max", "lambda_max"), ("orbitmin_trace", "MS_1"), ("orbitmin_sqrt_trace", "MS_1/2"),
        ("avg_filter_0.1", "filter 0.1"), ("avg_mult_0.1", "mult 0.1"), ("ggn_trace_wscaled", "sum w2 G"),
        ("test_acc", "test acc"), ("epoch", "epoch")]


def main(pairs):
    for main_name, interv_name in pairs:
        dm, di = load_sweep(main_name), load_sweep(interv_name)
        dm, di = dm[dm.target.notna()], di[di.target.notna()]
        for d in (dm, di):
            d["excess"] = d["ggn_trace"] / d["orbitmin_trace"]
        m = dm[dm.opt == "adam"].merge(di, on=["lr", "bs", "seed", "target"], suffixes=("_a", "_q"))
        matched = (np.log(m.train_loss_q / m.train_loss_a)).abs() < np.log(1.25)
        print(f"== {interv_name}: {di.run.nunique()} Adam-Q runs, {len(m)} paired checkpoints, "
              f"{matched.sum()} loss-matched within 25%")
        mm = m[matched]
        rows = []
        for col, lab in COLS:
            r = mm[f"{col}_q"] / mm[f"{col}_a"]
            rows.append({"measure": lab, "Adam (median)": mm[f"{col}_a"].median(),
                         "Adam-Q (median)": mm[f"{col}_q"].median(), "ratio Q/A median": r.median(),
                         "ratio q25": r.quantile(0.25), "ratio q75": r.quantile(0.75)})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        print("per target, median ratio Q/A:")
        print(mm.assign(r_tr=mm.ggn_trace_q / mm.ggn_trace_a, r_ms=mm.orbitmin_trace_q / mm.orbitmin_trace_a,
                        r_iso=mm["avg_iso_0.01_q"] / mm["avg_iso_0.01_a"], ex_a=mm.excess_a, ex_q=mm.excess_q)
              .groupby("target")[["r_tr", "r_ms", "r_iso", "ex_a", "ex_q"]].median().round(3))
        # Where would Adam-Q rank against SGD?
        both = pd.concat([dm[dm.opt.isin(["sgd", "sgdm", "sam", "adam"])], di.assign(opt="adamq")])
        for tgt in [0.3, 0.1, 0.01]:
            row = []
            for col in ["ggn_trace", "orig_rel_sharp_test", "avg_iso_0.01", "orbitmin_trace"]:
                pa, _ = prob_flatter(both, col, "adam", "sgd", tgt)
                pq, _ = prob_flatter(both, col, "adamq", "sgd", tgt)
                row.append(f"{col}: Adam {pa:.2f} -> Adam-Q {pq:.2f}")
            print(f"P(flatter than SGD) @ {tgt}: " + " | ".join(row))
        # training speed
        ea = dm[dm.opt == "adam"].groupby("run").epoch.max().median()
        eq = di.groupby("run").epoch.max().median()
        print(f"median epochs to finish: Adam {ea}, Adam-Q {eq}\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(list(zip(a[::2], a[1::2])))
