#!/usr/bin/env python3
"""
İkili modelin final değerlendirmesi: kilitli eşik, gri bölge, kalibrasyon, metrikler.

Eğitim döngüsünden ayrı tutulur ki (a) eğitim erken kesildiğinde de çalıştırılabilsin,
(b) düzeltilmiş gri bölge fonksiyonu kullanılsın.

Tüm geliştirme kararları YALNIZ validation setinde alınır: Platt kalibrasyonu val'de
fit edilir, karar eşiği val'de Youden ile seçilir, gri bölge sınırları val'de belirlenir.
Test setine hepsi değiştirilmeden uygulanır.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds                                                     # noqa: E402
from dataset import EchoClipDataset, compute_aux_stats, AUX_COLS         # noqa: E402
from model import DiastolicModel                                         # noqa: E402
from train_binary import (LAP, SPLITS, MANIFEST, study_probs, binary_metrics,   # noqa: E402
                          youden, gray_zone, fit_platt, apply_platt, calib_stats)


def murphy(y, p, bins=10):
    """Brier = güvenilirlik - çözünürlük + belirsizlik (Murphy 1973)."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, bins - 1)
    ybar = y.mean(); n = len(y); rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - ybar) ** 2
    unc = float(ybar * (1 - ybar))
    return {'guvenilirlik': round(rel / n, 5), 'cozunurluk': round(res / n, 5),
            'belirsizlik_taban': round(unc, 5)}


def full_metrics(y, p, thr=None):
    m = binary_metrics(y, p, thr)
    m.update(murphy(y, p))
    m['beceri_puani'] = round(1 - m['brier'] / m['belirsizlik_taban'], 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=os.path.expanduser('~/mimic-echo/runs/b2_binary'))
    ap.add_argument('--splits', default=SPLITS)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    a = ap.parse_args()
    dev = 'cuda'
    ds.GRADES[:] = LAP

    aux = compute_aux_stats(a.splits, MANIFEST)
    mk = lambda sp: EchoClipDataset(sp, a.clip_len, train=False, aux_stats=aux,
                                    splits_csv=a.splits, manifest_csv=MANIFEST)
    vl = DataLoader(mk('val'), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    tl = DataLoader(mk('test'), batch_size=a.batch, num_workers=a.workers, pin_memory=True)

    model = DiastolicModel(a.clip_len, 2, len(AUX_COLS), pretrained=True, dropout=0.25).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run_dir, 'best.pt'), weights_only=True))
    print('[model] best.pt yüklendi', flush=True)

    sv, yv, pv, _ = study_probs(model, vl, dev, a.aux_weight)
    st_, yt, pt, _ = study_probs(model, tl, dev, a.aux_weight)
    print(f'[veri] val {len(yv)} / test {len(yt)} çalışma', flush=True)

    ab = fit_platt(yv, pv)                                  # KALİBRASYON: yalnız val
    pv_c, pt_c = apply_platt(pv, ab), apply_platt(pt, ab)
    thr = youden(yv, pv_c)                                  # EŞİK: yalnız val
    lo, hi = gray_zone(yv, pv_c)                            # GRİ BÖLGE: yalnız val

    gz = (pt_c > lo) & (pt_c < hi)
    res = {
        'kaynak': 'egitim erken kesildi (val AUROC 3 epoch dusunce); best.pt = en iyi epoch',
        'kalibrasyon': {'yontem': 'Platt, yalniz val', 'a': round(ab[0], 4), 'b': round(ab[1], 4)},
        'esik': round(thr, 4), 'esik_kaynagi': 'yalniz val (Youden), kalibre olcekte',
        'gri_bolge': {'rule_out_alti': round(lo, 4), 'rule_in_ustu': round(hi, 4)},
        'val_ham': full_metrics(yv, pv), 'val': full_metrics(yv, pv_c, thr),
        'test_ham': full_metrics(yt, pt), 'test': full_metrics(yt, pt_c, thr),
        'kalibrasyon_val': calib_stats(yv, pv_c), 'kalibrasyon_test': calib_stats(yt, pt_c),
        'test_gri_bolge': {'belirsiz_n': int(gz.sum()), 'belirsiz_oran': round(float(gz.mean()), 4)},
    }
    if (~gz).sum() > 10:
        res['test_gri_disi'] = full_metrics(yt[~gz], pt_c[~gz], thr)
    json.dump(res, open(os.path.join(a.run_dir, 'binary_results.json'), 'w'), indent=1, ensure_ascii=False)
    for nm, sids, y, pc, praw in (('val', sv, yv, pv_c, pv), ('test', st_, yt, pt_c, pt)):
        with open(os.path.join(a.run_dir, f'{nm}_predictions.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['study_id', 'y_true', 'p_elevated', 'p_ham'])
            w.writerows([[s, int(yy), f'{c:.5f}', f'{r:.5f}'] for s, yy, c, r in zip(sids, y, pc, praw)])
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
