import sys, time, json
sys.path.insert(0, ".")
from lls.train import run
cfg = dict(dataset=sys.argv[1], arch=sys.argv[2], opt=sys.argv[3], lr=float(sys.argv[4]), bs=int(sys.argv[5]),
           seed=0, targets=[1.0, 0.1], max_epochs=int(sys.argv[6]), threads=int(sys.argv[7]), save_model=False)
r = run(cfg, verbose=True)
print(r["status"], r["time"])
ck = r["ckpts"][-1]
print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ck.items() if k != "layer_norms"})
