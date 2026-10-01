#!/usr/bin/env python3
"""
Ön-tanımlı ikincil analiz (analysis_plan §5.6): ön-işleme uyumu — histogram eşleme.

Plan yöntemi belirtmiyordu; operasyonel seçim (2026-10-01, analysis_log Ek 10):
  - Referans: geliştirme EĞİTİM bölümünden sabit tohumla 2.000 klip; her klipte sektör maskesi
    (zaman boyunca en büyük değer > 0) içindeki piksellerin gri düzey dağılımı, modele verilen
    16 karede (_sample_indices) toplanır.
  - Her dış klip: aynı maske içindeki dağılımı, klip bazında, referans kümülatif dağılıma eşlenir
    (256 düzeyli arama tablosu); maske dışı 0 kalır.
  - Model, Platt kalibratörü, eşik ve gri bölge kilitli, değişmez. Yalnız girdi yoğunluğu değişir.
Çıktı: external/external_predictions_histmatch.csv (run_external.py ile aynı sütunlar).
"""
from __future__ import annotations
import collections, csv, glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'grading'))
from dataset import _sample_indices, IMAGENET_MEAN, IMAGENET_STD, AUX_COLS  # noqa: E402
from model import DiastolicModel, corn_predict                              # noqa: E402
from train_binary import apply_platt                                        # noqa: E402

SEED = 20261001
RUN = os.path.expanduser('~/mimic-echo/runs/b2_binary')
DEV = os.path.expanduser('~/mimic-echo/npz_hires')
EXT = os.path.expanduser('~/echoxflow-render/npz')
VIEWS = os.path.expanduser('~/echoxflow-render/_views.csv')
OUT = os.path.join(HERE, 'external_predictions_histmatch.csv')


def masked_hist(fr):
    m = fr.max(0) > 0
    return np.bincount(fr[:, m].ravel(), minlength=256).astype(np.float64)


def reference_cdf(n=2000):
    train = {r['study_id'] for r in csv.DictReader(open(os.path.expanduser('~/mimic-echo/splits.csv')))
             if r['split'] == 'train'}
    files = [p for p in sorted(glob.glob(os.path.join(DEV, '*.npz'))) if os.path.basename(p).split('_')[0] in train]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    h = np.zeros(256)
    for k in pick:
        fr = np.load(files[k], allow_pickle=True)['frames']
        h += masked_hist(fr[_sample_indices(fr.shape[0], 16, False)])
    print(f'[referans] {len(pick)} eğitim klibi / {len(files)}', flush=True)
    return np.cumsum(h) / h.sum()


def match(fr, ref_cdf):
    m = fr.max(0) > 0
    h = np.bincount(fr[:, m].ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(h) / h.sum()
    lut = np.clip(np.searchsorted(ref_cdf, cdf), 0, 255).astype(np.uint8)
    out = lut[fr]; out[:, ~m] = 0
    return out


def main():
    ref_cdf = reference_cdf()
    L = json.load(open(os.path.join(RUN, 'binary_results.json')))
    ab = (L['kalibrasyon']['a'], L['kalibrasyon']['b'])
    keep = {r['dicom_id'] for r in csv.DictReader(open(VIEWS)) if r['keep'] == '1'}
    files = [p for p in sorted(glob.glob(os.path.join(EXT, '*.npz'))) if os.path.basename(p)[:-4] in keep]
    print(f'[veri] hedef görünümlü dış klip: {len(files)}', flush=True)
    model = DiastolicModel(16, 2, len(AUX_COLS), pretrained=True, dropout=0.25).cuda()
    model.load_state_dict(torch.load(os.path.join(RUN, 'best.pt'), weights_only=True)); model.eval()
    ssum, scnt = collections.defaultdict(float), collections.Counter()
    buf, meta = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ol, _ = model(torch.stack(buf).cuda())
        for ex, p in zip(meta, corn_predict(ol.float())[1].float().cpu().numpy()[:, 0]):
            ssum[ex] += float(p); scnt[ex] += 1
        buf.clear(); meta.clear()
    for i, fp in enumerate(files, 1):
        z = np.load(fp, allow_pickle=True)
        fr = match(z['frames'], ref_cdf)
        clip = fr[_sample_indices(fr.shape[0], 16, False)].astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)
        buf.append((x - IMAGENET_MEAN) / IMAGENET_STD); meta.append(str(z['exam_id']))
        if len(buf) >= 24:
            flush()
        if i % 500 == 0:
            print(f'  {i}/{len(files)}', flush=True)
    flush()
    base = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv')))}
    rows = []
    for e in sorted(ssum):
        if e not in base:
            continue
        praw = ssum[e] / scnt[e]
        r = dict(base[e]); r['p_ham'] = round(praw, 5)
        r['p_kalibre'] = round(float(apply_platt(np.array([praw]), ab)[0]), 5)
        rows.append(r)
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'-> {OUT} ({len(rows)} muayene; temel dosyada {len(base)})', flush=True)


if __name__ == '__main__':
    main()
