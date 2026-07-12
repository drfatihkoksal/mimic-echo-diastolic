#!/usr/bin/env python3
"""
Faz C — İkili (dikotom) değerlendirme: C.1 modelinden YENİDEN EĞİTİM OLMADAN.

Klinik gerekçe: "diyastolik disfonksiyon var mı?" doğrudan bir tarama sorusu; AUROC rapor eden
literatürle (Tromp 2021 E/e'≥13 AUC 0.91; Pandey 2021 yükselmiş dolum basıncı AUC 0.88) DOĞRUDAN
kıyaslanabilir. Grade3 seyrekliği sorununu da çözer.

CORN ordinal head ikili sınıflandırıcıyı zaten içerir: kümülatif olasılıklar
cum=[P(≥G1), P(≥G2), P(≥G3)]. İki klinik dikotomi:
  (1) herhangi-DD : Normal vs ≥Grade1  → skor = P(≥G1) = cum[0]
  (2) ileri-DD    : ≤Grade1 vs ≥Grade2 → skor = P(≥G2) = cum[1]  (≈yükselmiş LAP)

Girdi (--csv): test_predictions.csv (study_id,true,pred,confidence,n_clips,cum_probs) — C.1 formatı;
oof_predictions.csv (p_ge1/p_ge2/p_ge3) formatı da okunur.
Metrik: AUROC (+bootstrap %95 GA), AUPRC, Youden-eşiğinde sens/spec/acc/F1/PPV/NPV, prevalans,
        ayrıca QWK (4-sınıf ve 3-sınıf) için bootstrap %95 GA.

Kullanım:
  python3 binary_eval.py                                        # C.1 (varsayılan)
  python3 binary_eval.py --csv ~/mimic-echo/runs/c2_mil/mil_attention/test_predictions.csv \
                         --out ~/mimic-echo/runs/c2_mil/mil_attention/binary_metrics.json
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)

GRADES = ['Normal', 'Grade1', 'Grade2', 'Grade3']
DEFAULT_CSV = os.path.expanduser('~/mimic-echo/runs/c1_panecho/test_predictions.csv')


def load(path):
    rows = []
    with open(os.path.expanduser(path), newline='') as f:
        for r in csv.DictReader(f):
            if 'cum_probs' in r:
                cum = [float(v) for v in r['cum_probs'].split(';')]
            else:
                cum = [float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])]
            rows.append({'g': GRADES.index(r['true']), 'p': GRADES.index(r['pred']),
                         'cum': cum, 'conf': r.get('confidence', '')})
    return rows


def boot_qwk(t, p, n=2000, seed=42):
    """QWK bootstrap %95 GA (4-sınıf ve 3-sınıf birleşik)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(t)); k4, k3 = [], []
    t3, p3 = np.minimum(t, 2), np.minimum(p, 2)
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(t[b])) < 2:
            continue
        k4.append(cohen_kappa_score(t[b], p[b], weights='quadratic', labels=[0, 1, 2, 3]))
        k3.append(cohen_kappa_score(t3[b], p3[b], weights='quadratic', labels=[0, 1, 2]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {
        'QWK_4cls': round(float(cohen_kappa_score(t, p, weights='quadratic', labels=[0, 1, 2, 3])), 3),
        'QWK_4cls_95CI': q(k4),
        'QWK_3cls': round(float(cohen_kappa_score(t3, p3, weights='quadratic', labels=[0, 1, 2])), 3),
        'QWK_3cls_95CI': q(k3),
    }


def boot_auc(y, s, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate(rows, thr_grade, score_idx, subset=None):
    r = [x for x in rows if (subset is None or x['conf'] == subset)]
    y = np.array([1 if x['g'] >= thr_grade else 0 for x in r])
    s = np.array([x['cum'][score_idx] for x in r])
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s)
    ap = average_precision_score(y, s)
    # Youden-optimal eşik
    fpr, tpr, thr = roc_curve(y, s); j = np.argmax(tpr - fpr); t = thr[j]
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0; npv = tn / (tn + fn) if tn + fn else 0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0
    return {
        'n': len(r), 'prevalans': round(float(y.mean()), 3),
        'AUROC': round(auc, 3), 'AUROC_95CI': [round(lo, 3), round(hi, 3)],
        'AUPRC': round(float(ap), 3), 'esik': round(float(t), 3),
        'sens': round(sens, 3), 'spec': round(spec, 3), 'acc': round(float((p == y).mean()), 3),
        'PPV': round(ppv, 3), 'NPV': round(npv, 3), 'F1': round(f1, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default='~/mimic-echo/runs/c1_panecho/binary_metrics.json')
    args = ap.parse_args()

    rows = load(args.csv)
    print(f"Girdi: {args.csv} | n={len(rows)}")
    out = {}

    # QWK + bootstrap GA (genel ve ecg_sinus)
    for sub_name, sub in [('genel', None), ('ecg_sinus', 'ecg_sinus')]:
        r = [x for x in rows if (sub is None or x['conf'] == sub)]
        if len(r) < 20:
            continue
        t = np.array([x['g'] for x in r]); p = np.array([x['p'] for x in r])
        k = boot_qwk(t, p); k['n'] = len(r)
        out[f'QWK | {sub_name}'] = k
        print(f"\n=== QWK [{sub_name}] n={k['n']} ===")
        print(f"  4-sınıf: {k['QWK_4cls']} {k['QWK_4cls_95CI']} | 3-sınıf: {k['QWK_3cls']} {k['QWK_3cls_95CI']}")

    tasks = [('herhangi-DD (Normal vs ≥Grade1)', 1, 0),
             ('ileri-DD (≤Grade1 vs ≥Grade2)', 2, 1)]
    for name, thr, si in tasks:
        print(f"\n=== {name} ===")
        for sub_name, sub in [('genel', None), ('ecg_sinus', 'ecg_sinus')]:
            m = evaluate(rows, thr, si, sub)
            out[f'{name} | {sub_name}'] = m
            if m:
                print(f"[{sub_name:9s}] n={m['n']} prev={m['prevalans']} | "
                      f"AUROC={m['AUROC']} {m['AUROC_95CI']} AUPRC={m['AUPRC']} | "
                      f"sens={m['sens']} spec={m['spec']} acc={m['acc']} F1={m['F1']}")
    outp = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, 'w'), indent=1, ensure_ascii=False)
    print(f"\nKaydedildi: {outp}")


if __name__ == '__main__':
    main()
