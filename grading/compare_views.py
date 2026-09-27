#!/usr/bin/env python3
"""
Görünüm yapılandırması karşılaştırması — YALNIZ train/val.

Aşırı uyumun en doğrudan kaldıracı eğitim verisini artırmak. Faz A'da sınıflandırılmış
ama üretilmemiş A3C/A5C/PSAX klipleri yerel DICOM'lardan üretildi; bu betik üç
yapılandırmayı aynı hiperparametrelerle, aynı tohumla eğitir ve val eğrilerini yazar.

Karar ölçütü (sonuca bakılmadan önce sabitlendi):
  1) en yüksek val AUROC
  2) eşitlikte (fark < 0.005) daha GEÇ tepe yapan, yani daha az aşırı uyan

İç test seti ve dış kohort bu betikte HİÇ kullanılmaz.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

H = os.path.expanduser('~/mimic-echo/')
BASE = {'A4C', 'A2C', 'Parasternal_Long'}
CONFIGS = {
    'A_temel':      sorted(BASE),
    'B_apikal_tam': sorted(BASE | {'A3C', 'A5C'}),
    'C_tum':        sorted(BASE | {'A3C', 'A5C', 'PSAX'}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    out = {}
    for name, views in CONFIGS.items():
        if a.only and name != a.only:
            continue
        d = os.path.join(H, 'runs', f'views_{name}')
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_binary.py'),
               '--out-dir', d, '--epochs', str(a.epochs), '--patience', str(a.patience),
               '--views', ','.join(views), '--extra-dirs', os.path.join(H, 'npz_views')]
        print(f'\n=== {name}: {views} ===', flush=True)
        log = os.path.join(H, f'views_{name}.log')
        with open(log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        hist = []
        for line in open(log):
            if line.startswith('[ep'):
                try:
                    ep = int(line.split(']')[0][3:])
                    au = float(line.split('AUROC=')[1].split()[0])
                    hist.append({'ep': ep, 'auroc': au})
                except Exception:
                    pass
        if hist:
            best = max(hist, key=lambda x: x['auroc'])
            out[name] = {'views': views, 'n_epoch': len(hist), 'best_ep': best['ep'],
                         'best_auroc': best['auroc'], 'hist': hist}
            print(f"  {name}: en iyi val AUROC={best['auroc']} (ep{best['ep']}, {len(hist)} epoch)", flush=True)
    p = os.path.join(H, 'views_comparison.json')
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(out); json.dump(prev, open(p, 'w'), indent=1, ensure_ascii=False)
    if len(prev) > 1:
        rank = sorted(prev.items(), key=lambda kv: (-kv[1]['best_auroc'], -kv[1]['best_ep']))
        top = rank[0]
        close = [k for k, v in prev.items() if top[1]['best_auroc'] - v['best_auroc'] < 0.005]
        if len(close) > 1:
            top = max(((k, prev[k]) for k in close), key=lambda kv: kv[1]['best_ep'])
        print(f"\nSEÇİM: {top[0]}  (AUROC {top[1]['best_auroc']}, tepe ep{top[1]['best_ep']})", flush=True)
    print(json.dumps(prev, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
