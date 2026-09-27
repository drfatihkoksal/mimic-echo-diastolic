#!/usr/bin/env python3
"""
Dış validasyon koşusu — EchoXFlow (Akershus, Norveç), SAF ZERO-SHOT.

Hiçbir yeniden eğitim, hiçbir yeniden ayar yok. İç kohortta kilitlenmiş dört şey
olduğu gibi uygulanır:
  1. model ağırlıkları (b2_binary/best.pt, val AUROC ile seçilen epoch)
  2. Platt kalibratörü (yalnız iç val'de fit edilmişti)
  3. karar eşiği (yalnız iç val'den, Youden)
  4. gri bölge sınırları (yalnız iç val'den)

Ön-işleme iç boru hattıyla birebir: (T,224,224) uint8 -> /255 -> 3 kanal -> ImageNet norm,
klip boyunca uniform 16 kare (deterministik). Klip olasılıkları muayene düzeyinde ortalanır.

Son noktalar (external/PRESPECIFICATION.md'de sonuçlar görülmeden kilitlendi):
  birincil : yüksek E/e' (septal>=15 | lateral>=13 | ort>=14)
  ikincil  : ASE 2-kriterli LAP (algoritmanın belirlenim dalları)
  duyarlılık: 3/3 birincil mevcut, çoğunluk kuralı
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'grading'))
from dataset import _sample_indices, IMAGENET_MEAN, IMAGENET_STD, AUX_COLS  # noqa: E402
from model import DiastolicModel, corn_predict                              # noqa: E402
from train_binary import binary_metrics, apply_platt                        # noqa: E402
from finalize_binary import murphy                                          # noqa: E402
from diastolic_grading import derive, ee_value                              # noqa: E402


def load_clip(fp, clip_len=16):
    z = np.load(fp, allow_pickle=True)
    fr = z['frames']
    idx = _sample_indices(fr.shape[0], clip_len, False)
    clip = fr[idx].astype(np.float32) / 255.0
    x = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD, str(z['exam_id'])


def f(r, k):
    v = r.get(k)
    return float(v) if v not in ('', 'None', None) else None


def endpoints(ref_row):
    """Ön-tanımlı son noktalar. Döner: dict(ad -> 0/1/None)."""
    p = derive({'E': f(ref_row, 'E'), 'e_sep': f(ref_row, 'e_sep'), 'e_lat': f(ref_row, 'e_lat'),
                'TRvel': f(ref_row, 'TRvel'), 'EA': f(ref_row, 'EA')})
    checks = [(ee_value(p, 'sep'), 15), (ee_value(p, 'lat'), 13), (ee_value(p, 'mean'), 14)]
    ee_av = any(v is not None for v, _ in checks)
    ee_hi = any(v is not None and v >= t for v, t in checks)
    out = {'birincil_ee': (int(ee_hi) if ee_av else None)}
    lap = ref_row.get('lap')
    out['ikincil_lap'] = (1 if lap == 'elevated' else 0 if lap == 'normal' else None)
    n_av = int(ref_row['n_primary_available']) if ref_row.get('n_primary_available') not in ('', None) else 0
    n_ab = ref_row.get('n_abnormal')
    out['duyarlilik_cogunluk'] = (int(int(n_ab) >= 2) if (n_av == 3 and n_ab not in ('', None)) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=os.path.expanduser('~/mimic-echo/runs/b2_binary'))
    ap.add_argument('--npz-dir', default=os.path.expanduser('~/echoxflow-render/npz'))
    ap.add_argument('--views', default=os.path.expanduser('~/echoxflow-render/_views.csv'))
    ap.add_argument('--ref', default=os.path.join(HERE, 'echoxflow_reference_v2.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'external_results.json'))
    ap.add_argument('--batch', type=int, default=24)
    a = ap.parse_args()
    dev = 'cuda'

    locked = json.load(open(os.path.join(a.run_dir, 'binary_results.json')))
    ab = (locked['kalibrasyon']['a'], locked['kalibrasyon']['b'])
    thr = locked['esik']; lo = locked['gri_bolge']['rule_out_alti']; hi = locked['gri_bolge']['rule_in_ustu']
    print(f'[kilitli] Platt a={ab[0]} b={ab[1]} | eşik={thr} | gri bölge=({lo}, {hi})', flush=True)

    keep = {r['dicom_id'] for r in csv.DictReader(open(a.views)) if r['keep'] == '1'}
    files = [p for p in sorted(glob.glob(os.path.join(a.npz_dir, '*.npz')))
             if os.path.basename(p)[:-4] in keep]
    print(f'[veri] hedef görünümlü klip: {len(files)}', flush=True)

    model = DiastolicModel(16, 2, len(AUX_COLS), pretrained=True, dropout=0.25).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run_dir, 'best.pt'), weights_only=True))
    model.eval()

    ssum, scnt = collections.defaultdict(float), collections.Counter()
    buf, meta = [], []
    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ol, _ = model(x)
        pr = corn_predict(ol.float())[1].float().cpu().numpy()[:, 0]
        for ex, p in zip(meta, pr):
            ssum[ex] += float(p); scnt[ex] += 1
        buf.clear(); meta.clear()
    for i, fp in enumerate(files, 1):
        x, ex = load_clip(fp)
        buf.append(x); meta.append(ex)
        if len(buf) >= a.batch:
            flush()
        if i % 500 == 0:
            print(f'  {i}/{len(files)}', flush=True)
    flush()
    praw = {e: ssum[e] / scnt[e] for e in ssum}
    pcal = {e: float(apply_platt(np.array([praw[e]]), ab)[0]) for e in praw}
    print(f'[çıkarım] {len(praw)} muayene', flush=True)

    ref = {r['exam_id']: r for r in csv.DictReader(open(a.ref))}
    res = {'kilitli': {'platt': ab, 'esik': thr, 'gri_bolge': [lo, hi]},
           'klip': len(files), 'muayene': len(praw), 'son_noktalar': {}}
    rows = []
    for name in ('birincil_ee', 'ikincil_lap', 'duyarlilik_cogunluk'):
        ys, ps, exs = [], [], []
        for ex, r in ref.items():
            if ex not in pcal or int(r['af_excluded']):
                continue
            y = endpoints(r)[name]
            if y is None:
                continue
            ys.append(y); ps.append(pcal[ex]); exs.append(ex)
        y = np.array(ys); p = np.array(ps)
        if len(y) < 20 or y.sum() < 5:
            res['son_noktalar'][name] = {'n': int(len(y)), 'not': 'yetersiz'}
            continue
        m = binary_metrics(y, p, thr); m.update(murphy(y, p))
        m['beceri_puani'] = round(1 - m['brier'] / m['belirsizlik_taban'], 4)
        gz = (p > lo) & (p < hi)
        m['gri_bolge'] = {'belirsiz_n': int(gz.sum()), 'belirsiz_oran': round(float(gz.mean()), 4)}
        if (~gz).sum() > 20 and y[~gz].sum() >= 5:
            g = binary_metrics(y[~gz], p[~gz], thr)
            m['gri_disi'] = {k: g[k] for k in ('n', 'olay', 'auroc', 'duyarlilik', 'ozgulluk', 'PPD', 'NPD')}
        if name == 'birincil_ee':
            terts = json.load(open(os.path.join(HERE, 'clipping_tertiles.json')))['tersil_sinirlari']
            cl = np.array([f(ref[e], 'clip_primary') if f(ref[e], 'clip_primary') is not None else np.nan for e in exs])
            band = np.digitize(cl, terts)
            m['kirpilma_tersilleri'] = []
            for t in (0, 1, 2):
                sel = (band == t) & ~np.isnan(cl)
                if sel.sum() >= 20 and y[sel].sum() >= 5:
                    from sklearn.metrics import roc_auc_score
                    m['kirpilma_tersilleri'].append(
                        {'tersil': t + 1, 'n': int(sel.sum()), 'olay': int(y[sel].sum()),
                         'auroc': round(float(roc_auc_score(y[sel], p[sel])), 4)})
        res['son_noktalar'][name] = m

    # TÜM muayenelerin tahminleri yazılır (yalnız birincil son noktası tanımlı olanlar değil):
    # 4 muayenenin ASE LAP'i belirli ama E/e' hesaplanamiyor; onlar da ikincil kohorta girer.
    for e in sorted(pcal):
        r = ref.get(e)
        if r is None:
            continue
        ep = endpoints(r)
        rows.append({'exam_id': e,
                     'y_birincil': ('' if ep['birincil_ee'] is None else int(ep['birincil_ee'])),
                     'y_ikincil': ('' if ep['ikincil_lap'] is None else int(ep['ikincil_lap'])),
                     'af_excluded': int(r['af_excluded']),
                     'p_kalibre': round(pcal[e], 5), 'p_ham': round(praw[e], 5),
                     'clip_primary': r['clip_primary']})
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, 'external_predictions.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
