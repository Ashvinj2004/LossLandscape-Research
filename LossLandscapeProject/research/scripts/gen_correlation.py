"""Rank correlation (Kendall tau) between each measure and generalization, at matched training loss."""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lls.analysis import load_sweep
from analyze_main import MEASURES

def table(name, targets=(0.1, 0.03, 0.01), target_col="test_loss"):
    df = load_sweep(name); df = df[df["target"].notna()]
    rows = []
    for mname, (col, fam) in MEASURES.items():
        if col not in df: continue
        row = {"measure": mname, "family": fam}
        for t in targets:
            sub = df[df["target"] == t][[col, target_col]].dropna()
            tau, p = kendalltau(sub[col], sub[target_col])
            row[f"tau@{t}"] = round(tau, 2)
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for name in sys.argv[1:]:
        print(f"== {name}: Kendall tau(measure, test loss) pooled over optimizers")
        print(table(name).to_string(index=False))
        print(f"== {name}: Kendall tau(measure, test error)")
        df = load_sweep(name); 
        print(table(name, target_col="test_err" if False else "gap_acc").to_string(index=False))
