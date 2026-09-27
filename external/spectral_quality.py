#!/usr/bin/env python3
"""
Spektral kayıt kalitesi bayrağı: hız ölçeği dar seçildiğinde Doppler zarfı kadrajın
kenarında kesilir (clipping). Yazarın kör panel değerlendirmesinde "yanlış" bulunan
13 izin tamamı bu özellikle açıklanıyor (kenar-sinyal oranı medyan 0,712; sağlam
izlerde 0,074).

Ölçüt: matrisin hız ekseni boyunca en üst/en alt %2'lik bantlarında, 60. yüzdeliğin
üstünde sinyal taşıyan piksellerin oranı. Zarf kadraja sığıyorsa bu oran düşüktür.
"""
from __future__ import annotations
import csv, glob, json, os, sys, tarfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'echoxflow-code', 'src'))
import zarr  # noqa: E402

PANEL = {'MV E/A': 'MV E/A', 'Eprime Septal': 'e_sep', 'Eprime Lateral': 'e_lat', 'TR Vmax': 'TR'}


def trace_quality(mat, zero_row, flow_sign, edge_frac=0.03, guard=0.20, pct=60):
    """Kırpılma: AKIŞIN GÖSTERİLDİĞİ taraftaki en dış bantta sinyal oranı.

    İLK SÜRÜMDEKİ HATA: bant hız ekseninin her iki ucunda ölçülüyordu. Operatör taban
    çizgisini kenara kaydırdığında (baseline_frac 0,1 veya 0,9 — tamamen normal bir ayar)
    sıfır-hız clutter bandı o kenara düşüyor ve ölçütü şişiriyordu. Ölçülen şey kırpılma
    değil taban çizgisinin konumuydu: tabanın kenara uzaklığı ile eski ölçüt arasında
    Pearson r = -0,73; bf=0,5'te medyan 0,041, bf=0,9'da 0,784.

    Düzeltilmiş hali yalnız akış tarafındaki en dış bandı ölçer ve bant taban çizgisine
    guard*N'den yakınsa ölçümü tanımsız bırakır. Yazarın kör değerlendirmesinde sağlam
    izlerde medyan 0,069, sorunlu izlerde 0,585.
    """
    N = mat.shape[1]
    e = max(2, int(edge_frac * N))
    thr = np.percentile(mat, pct)
    if flow_sign > 0:
        band, center = mat[:, :e], e / 2.0
    else:
        band, center = mat[:, -e:], N - e / 2.0
    if abs(zero_row - center) < guard * N:
        return None, float((mat > 240).mean())
    return float((band > thr).mean()), float((mat > 240).mean())


def scan(tar_dir='echoxflow/exams', out='external/spectral_quality.csv', tmp=None):
    tmp = tmp or os.path.expanduser('~/echoxflow-render/_qtmp')
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for i, tp in enumerate(sorted(glob.glob(os.path.join(tar_dir, '*.tar'))), 1):
        ex = os.path.basename(tp)[:-4]
        want = {}
        with tarfile.open(tp) as tf:
            for m in tf:
                if m.name.endswith('.zarr/.zattrs'):
                    d = json.load(tf.extractfile(m))
                    for t in d.get('recording_manifest', {}).get('tracks') or []:
                        for a in (t.get('spectral_annotations') or []):
                            for k in PANEL:
                                if k in (a.get('label') or ''):
                                    want.setdefault(d['recording_id'], set()).add(k)
            if not want:
                continue
            pats = tuple(f'{r}.zarr/' for r in want)
            tf.extractall(tmp, members=[m for m in tf if any(p in m.name for p in pats)], filter='data')
        for rid, kinds in want.items():
            zp = os.path.join(tmp, 'exams', ex, f'{rid}.zarr')
            try:
                att = json.load(open(zp + '/.zattrs')); z = zarr.open(zp, mode='r')
                for t in att['recording_manifest'].get('tracks') or []:
                    sa = t.get('spectral_annotations') or []
                    if not sa:
                        continue
                    m = np.asarray(z[t['data']['zarr_path']])
                    zero = t['spectral_row_baseline_frac'] * m.shape[1]
                    vs = [q['velocity_mps'] for a in sa for q in a['points']]
                    sign = 1 if float(np.median(vs)) > 0 else -1
                    clip, sat = trace_quality(m, zero, sign)
                    if clip is None:
                        clip = -1     # tanimsiz
                    for k in kinds:
                        rows.append({'exam_id': ex, 'recording_id': rid, 'panel': PANEL[k],
                                     'edge_ratio': round(clip, 4), 'saturation': round(sat, 5)})
            except Exception as e:
                rows.append({'exam_id': ex, 'recording_id': rid, 'panel': 'ERR',
                             'edge_ratio': -1, 'saturation': str(e)[:40]})
        import shutil; shutil.rmtree(os.path.join(tmp, 'exams', ex), ignore_errors=True)
        if i % 100 == 0:
            print(i, len(rows), flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exam_id', 'recording_id', 'panel', 'edge_ratio', 'saturation'])
        w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} iz -> {out}')


if __name__ == '__main__':
    scan()
