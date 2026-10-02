import json, glob, os, sys
pat = sys.argv[1]
for p in sorted(glob.glob(pat)):
    r = json.load(open(p)); h = r['history']
    pts = [h[i] for i in [0, 9, 49, 99, 199, len(h) - 1] if i < len(h)]
    print(os.path.basename(p)[:-5], r['status'], ' | '.join(f'ep{e}:{l:.3f}/{a:.2f}' for e, s, l, a in pts))
    for c in r['ckpts']:
        if 'lambda_max' not in c:
            print(f"    tgt {c['target']} ep {c['epoch']} trL {c['train_loss']:.3f} teAcc {c['test_acc']:.3f}"); continue
        print(f"    tgt {c['target']} ep {c['epoch']} trL {c['train_loss']:.3f} teAcc {c['test_acc']:.3f} lam {c['lambda_max']:.1f} "
              f"lamP {c.get('lambda_max_precond', float('nan')):.0f} tr {c['hess_trace']:.1f} iso.01 {c['avg_iso_0.01']:.4f} "
              f"rel_orig {c['orig_rel_sharp_test']:.4f} mult.03 {c['avg_mult_0.03']:.4f} wn {c['w_norm']:.1f}")
