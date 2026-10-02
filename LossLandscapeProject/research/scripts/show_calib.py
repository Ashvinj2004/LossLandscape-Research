import json, glob, sys
for p in sorted(glob.glob(f'results/{sys.argv[1]}/*.json')):
    r = json.load(open(p)); c = r['cfg']
    reach = {ck['target']: ck['epoch'] for ck in r['ckpts']}
    print(f"{c['opt']:5s} lr={c['lr']:<7} bs={c['bs']:<4} {r['status']:10s} ep={r['epochs']:3d} t={r['time']:4.0f}s "
          f"final_loss={r['history'][-1][2]:.3f} reach={reach} acc={[round(ck['test_acc'], 3) for ck in r['ckpts']]}")
