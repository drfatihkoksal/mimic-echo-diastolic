#!/usr/bin/env python3
"""
İki modeli AYNI test study'leri üzerinde eşleştirilmiş bootstrap ile kıyasla.

Neden gerekli: C.1 (0.515) ve C.2 MIL (0.537) QWK'lerinin bootstrap güven aralıkları büyük
ölçüde örtüşüyor. Ayrı ayrı GA'lara bakıp "MIL daha iyi" demek geçersiz — aynı test setinde
aynı study'ler değerlendirildiği için FARKIN kendisi eşleştirilmiş olarak bootstrap'lanmalı
(korelasyonu hesaba katar, çok daha dar GA verir).

Çıktı: ~/mimic-echo/runs/model_comparison.json

Kullanım:
  python3 compare_models.py --a runs/c1_panecho/test_predictions.csv \
                            --b runs/c2_mil/mil_attention/test_predictions.csv \
                            --name-a C.1-PanEcho --name-b C.2-MIL
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

GRADES = ['Normal', 'Grade1', 'Grade2', 'Grade3']
R = os.path.expanduser('~/mimic-echo/')


def load(path):
    d = {}
    with open(os.path.expanduser(path) if path.startswith('~') else
              (path if os.path.isabs(path) else os.path.join(R, path)), newline='') as f:
        for r in csv.DictReader(f):
            cum = ([float(v) for v in r['cum_probs'].split(';')] if 'cum_probs' in r
                   else [float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])])
            d[r['study_id']] = (GRADES.index(r['true']), GRADES.index(r['pred']), cum)
    return d


def qwk(t, p):
    return cohen_kappa_score(t, p, weights='quadratic', labels=[0, 1, 2, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='runs/c1_panecho/test_predictions.csv')
    ap.add_argument('--b', default='runs/c2_mil/mil_attention/test_predictions.csv')
    ap.add_argument('--name-a', default='C.1-PanEcho')
    ap.add_argument('--name-b', default='C.2-MIL')
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--out', default='~/mimic-echo/runs/model_comparison.json')
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    sids = sorted(set(A) & set(B))
    t = np.array([A[s][0] for s in sids])
    pa = np.array([A[s][1] for s in sids]); pb = np.array([B[s][1] for s in sids])
    # kümülatif skorlar: P(≥G1)=cum[0] (herhangi-DD), P(≥G2)=cum[1] (ileri-DD)
    sa1 = np.array([A[s][2][0] for s in sids]); sb1 = np.array([B[s][2][0] for s in sids])
    sa2 = np.array([A[s][2][1] for s in sids]); sb2 = np.array([B[s][2][1] for s in sids])
    y_any = (t >= 1).astype(int); y_adv = (t >= 2).astype(int)

    obs = {
        'QWK_4cls': (qwk(t, pa), qwk(t, pb)),
        'AUROC_anyDD': (roc_auc_score(y_any, sa1), roc_auc_score(y_any, sb1)),
        'AUROC_advDD': (roc_auc_score(y_adv, sa2), roc_auc_score(y_adv, sb2)),
    }

    rng = np.random.default_rng(42); idx = np.arange(len(sids))
    diffs = {k: [] for k in obs}
    for _ in range(args.n_boot):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(t[bi])) < 2:
            continue
        diffs['QWK_4cls'].append(qwk(t[bi], pb[bi]) - qwk(t[bi], pa[bi]))
        if len(np.unique(y_any[bi])) > 1:
            diffs['AUROC_anyDD'].append(roc_auc_score(y_any[bi], sb1[bi]) - roc_auc_score(y_any[bi], sa1[bi]))
        if len(np.unique(y_adv[bi])) > 1:
            diffs['AUROC_advDD'].append(roc_auc_score(y_adv[bi], sb2[bi]) - roc_auc_score(y_adv[bi], sa2[bi]))

    out = {'n_test': len(sids), 'model_a': args.name_a, 'model_b': args.name_b,
           'n_boot': args.n_boot, 'metrics': {}}
    print(f"n={len(sids)} | {args.name_b} − {args.name_a} (eşleştirilmiş bootstrap, {args.n_boot}×)\n")
    for k, (va, vb) in obs.items():
        d = np.array(diffs[k]); lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))   # iki-yönlü bootstrap p
        out['metrics'][k] = {args.name_a: round(float(va), 3), args.name_b: round(float(vb), 3),
                             'delta': round(float(vb - va), 3),
                             'delta_95CI': [round(float(lo), 3), round(float(hi), 3)],
                             'p_boot': round(p, 4), 'significant': bool(p < 0.05)}
        print(f"  {k:12s}: {va:.3f} → {vb:.3f} | Δ={vb-va:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f} "
              f"{'✓ anlamlı' if p < 0.05 else '✗ anlamlı değil'}")

    outp = os.path.expanduser(args.out)
    json.dump(out, open(outp, 'w'), indent=1, ensure_ascii=False)
    print(f"\nKaydedildi: {outp}")


if __name__ == '__main__':
    main()
