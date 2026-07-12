#!/usr/bin/env python3
"""
Faz C — External baseline: PanEcho'nun KENDİ LVDiastolicFunction head'i (sıfır eğitim).

Amaç: (1) güçlü hazır karşılaştırma, (2) 'gap' kanıtı — PanEcho 3-sınıf kendi şemasında
(Normal / Mild|Indeterminate / Moderate|Severe) tahmin eder; bu bizim 3-sınıf birleşik
ölçeğimize (Normal / Grade1 / Grade2-3) BİREBİR denk gelir → AYNI test study'lerinde head-to-head.

PanEcho çıktı sırası: [Mild|Indeterminate, Moderate|Severe, Normal] → ordinal eşleme [1,2,0].
Ground-truth: grade_2025 (0=N,1=G1,2=G2,3=G3) → 3-sınıf ordinal min(grade,2).
Study-level: klip softmax olasılıkları study içinde ortalanır → argmax → ordinal.

Kullanım: python3 panecho_baseline.py [--split test] [--out-dir ~/mimic-echo/runs/panecho_baseline]
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, confusion_matrix, f1_score

from dataset import EchoClipDataset, compute_aux_stats

ORD3 = ['Normal', 'Grade1', 'Grade2-3']          # 3-sınıf ordinal
CLASS_TO_ORD = [1, 2, 0]                           # PanEcho class_names sırası → ordinal


def metrics3(recs, subset=None):
    r = [x for x in recs if (subset is None or x['confidence'] == subset)]
    if len(r) < 2: return None
    t = np.array([x['true3'] for x in r]); p = np.array([x['pred3'] for x in r])
    lab = [0, 1, 2]
    return {
        'n': len(r),
        'qwk': round(float(cohen_kappa_score(t, p, weights='quadratic', labels=lab)), 4),
        'acc': round(float((t == p).mean()), 4),
        'acc_pm1': round(float((np.abs(t - p) <= 1).mean()), 4),
        'bacc': round(float(balanced_accuracy_score(t, p)), 4),
        'macro_f1': round(float(f1_score(t, p, average='macro', labels=lab, zero_division=0)), 4),
        'confusion': confusion_matrix(t, p, labels=lab).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--out-dir', default=os.path.expanduser('~/mimic-echo/runs/panecho_baseline'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'

    model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', tasks=['LVDiastolicFunction'],
                           pretrained=True, clip_len=args.clip_len, trust_repo=True).eval().to(dev)
    ds = EchoClipDataset(args.split, args.clip_len, train=False, aux_stats=compute_aux_stats())
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"[panecho-baseline] {args.split}: {len(ds)} klip", flush=True)

    prob_sum = defaultdict(lambda: np.zeros(3)); cnt = defaultdict(int)
    true_of, conf_of = {}, {}
    with torch.no_grad():
        for b in dl:
            x = b['x'].to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(x)['LVDiastolicFunction'].float().cpu().numpy()   # (B,3) softmax
            for i, sid in enumerate(b['study_id']):
                prob_sum[sid] += p[i]; cnt[sid] += 1
                true_of[sid] = min(int(b['grade'][i]), 2)                   # 3-sınıf ordinal GT
                conf_of[sid] = b['confidence'][i]

    recs = []
    for sid in prob_sum:
        mp = prob_sum[sid] / cnt[sid]                                       # ortalama olasılık
        pred3 = CLASS_TO_ORD[int(mp.argmax())]
        recs.append({'study_id': sid, 'true3': true_of[sid], 'pred3': pred3,
                     'confidence': conf_of[sid], 'n_clips': cnt[sid],
                     'probs': ';'.join(f'{v:.3f}' for v in mp)})

    M = metrics3(recs); Meog = metrics3(recs, 'ecg_sinus')
    with open(os.path.join(args.out_dir, f'{args.split}_predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['study_id', 'true3', 'pred3', 'confidence', 'n_clips', 'probs_Mild_ModSev_Normal'])
        for r in recs:
            w.writerow([r['study_id'], ORD3[r['true3']], ORD3[r['pred3']], r['confidence'], r['n_clips'], r['probs']])
    json.dump({'all': M, 'ecg_sinus': Meog}, open(os.path.join(args.out_dir, f'{args.split}_metrics.json'), 'w'), indent=1)

    print("\n=== PanEcho external baseline — 3-sınıf (Normal/Grade1/Grade2-3) ===")
    for name, m in [('genel', M), ('ecg_sinus', Meog)]:
        if m: print(f"[{name}] n={m['n']} QWK={m['qwk']} acc={m['acc']} ±1={m['acc_pm1']} bacc={m['bacc']} macroF1={m['macro_f1']}")
    print("confusion (gerçek↓/tahmin→) [N,G1,G2-3]:")
    for i, row in enumerate(M['confusion']): print(f"  {ORD3[i]:>9}: {row}")


if __name__ == '__main__':
    main()
