"""Load sweep JSONs into a tidy checkpoint-level DataFrame."""
import glob
import json
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OPT_LABEL = {"sgd": "SGD", "sgdm": "SGD+M", "adam": "Adam", "sam": "SAM", "adamq": "Adam-Q"}


def load_sweep(name):
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, name, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        cfg = r["cfg"]
        for ck in r["ckpts"]:
            row = {k: v for k, v in cfg.items() if k not in ("targets",)}
            row["status"] = r["status"]
            row["run"] = os.path.basename(p)[:-5]
            row.update({k: v for k, v in ck.items() if not isinstance(v, list)})
            for i, ln in enumerate(ck.get("layer_norms", [])):
                row[f"layer_norm_{i}"] = ln
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["gap_loss"] = df["test_loss"] - df["train_loss"]
        df["gap_acc"] = df["train_acc"] - df["test_acc"]
        df["opt_label"] = df["opt"].map(OPT_LABEL)
    return df


def final_ckpts(df):
    return df.sort_values("epoch").groupby("run").tail(1)
