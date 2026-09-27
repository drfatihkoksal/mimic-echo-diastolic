#!/usr/bin/env python3
"""
Backbone karşılaştırması — YALNIZ train/val.

Sorulan soru dar tutuldu: backbone seçimi, düzeltmeye çalıştığımız aşırı uyumu
etkiliyor mu? "Daha iyi bir temel model var mı" sorusu sonu gelmez ve her aday dış
kohortu bir kez daha kullanmayı gerektirir.

Üç kol:
  P_ince_ayar : PanEcho, ince ayarlı            (mevcut yapılandırma, referans)
  P_dondurulmus: PanEcho, tamamen donuk + başlık (kapasite kontrolü — aşırı uyan
                 kısım ince ayar mı?)
  R_kinetics  : R(2+1)D-18, Kinetics ön-eğitimli (ön-eğitim ALANI kontrolü — eko-özgü
                 ön-eğitim mi kazandırıyor, yoksa herhangi bir video ön-eğitimi mi?)

Görünüm kümesi, görünüm karşılaştırmasının kazananıdır; komut satırından verilir.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

H = os.path.expanduser('~/mimic-echo/')
# Referans kol (PanEcho ince ayarlı, 3 görünüm, freeze 3) AYRICA KOŞULMAZ:
# görünüm karşılaştırmasındaki A_temel tam olarak o yapılandırmadır ve aynı
# epoch/sabır bütçesiyle koşulmuştur (en iyi val AUROC 0.8562, ep4). Sayı
# views_comparison.json'dan okunur; aynı eğitimi tekrarlamak GPU israfıdır.
# P_dondurulmus da KOŞULMAZ: ilk 3 epoch'u A_temel ile birebir aynıdır (o da ep0-2'de
# backbone'u donuk tutar) ve o üç epoch boyunca val AUROC 0.8418 -> 0.8381 ile düşüyordu.
# Donuk backbone tavanı 0.8418; ince ayarlı 0.8562. Kısmi sonuç kaydedildi, kol kesildi.
ARMS = {
    'R_kinetics': ['--arch', 'r2plus1d', '--freeze-epochs', '3'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--patience', type=int, default=5)
    a = ap.parse_args()
    out = {}
    for name, extra in ARMS.items():
        d = os.path.join(H, 'runs', f'bb_{name}')
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_binary.py'),
               '--out-dir', d, '--epochs', str(a.epochs), '--patience', str(a.patience),
               '--views', a.views, '--extra-dirs', os.path.join(H, 'npz_views')] + extra
        print(f'\n=== {name} ===', flush=True)
        log = os.path.join(H, f'bb_{name}.log')
        with open(log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        hist = []
        for line in open(log):
            if line.startswith('[ep'):
                try:
                    hist.append({'ep': int(line.split(']')[0][3:]),
                                 'auroc': float(line.split('AUROC=')[1].split()[0])})
                except Exception:
                    pass
        if hist:
            b = max(hist, key=lambda x: x['auroc'])
            out[name] = {'n_epoch': len(hist), 'best_ep': b['ep'], 'best_auroc': b['auroc'], 'hist': hist}
            print(f"  {name}: en iyi val AUROC={b['auroc']} (ep{b['ep']}/{len(hist)})", flush=True)
    pj = os.path.join(H, 'backbone_comparison.json')
    if os.path.exists(pj):                       # önceki kolları koru, üzerine yazma
        prev = json.load(open(pj)); prev.update(out); out = prev
    ref = json.load(open(os.path.join(H, 'views_comparison.json')))['A_temel']
    out['P_ince_ayar_referans'] = {'kaynak': 'views_comparison.json:A_temel (yeniden koşulmadı)',
                                   'n_epoch': ref['n_epoch'], 'best_ep': ref['best_ep'],
                                   'best_auroc': ref['best_auroc'], 'hist': ref['hist']}
    json.dump(out, open(os.path.join(H, 'backbone_comparison.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
