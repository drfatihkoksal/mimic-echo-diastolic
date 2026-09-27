#!/usr/bin/env python3
"""
Dış validasyon — Adım 2: EchoXFlow B-mode kayıtlarının render'ı.

Her muayenenin tar'ı açılır, SAF 2d_brightness_mode kayıtları beamspace'ten
kartezyene çevrilir (echoxflow.scan), ultrason sektörüne kırpılır, kareye
pad'lenir ve 224x224'e indirilir — MIMIC tarafındaki `redownload_hires.py`
ile BİREBİR aynı geometrik işlem, böylece iki kohort arasındaki tek fark
görüntünün kendisi olur.

Hız: sektör->kartezyen örnekleme koordinatları kayıt başına BİR KEZ hesaplanıp
cv2.remap ile tüm karelere uygulanır.

Çıktı: <out>/npz/<exam_id>__<recording_id>.npz  (frames=(T,224,224) uint8, meta)
       <out>/render_manifest.csv
Muayene işlendikten sonra açılan dosyalar silinir (disk sabit kalır).
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys, tarfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'echoxflow-code', 'src'))
import cv2, zarr                                                  # noqa: E402
from echoxflow.scan.geometry import sector_geometry_from_mapping  # noqa: E402
from echoxflow.scan.conversion import _sector_sample_coordinates  # noqa: E402
from echoxflow.scan.geometry import CartesianGrid                 # noqa: E402

RENDER_HEIGHT = 448          # kartezyen ara çözünürlük (224'ün 2 katı; aliasing'i önler)
SIZE = 224                   # modele giden boyut


def bmode_recordings(tar_path):
    """tar içindeki saf 2d_brightness_mode kayıtları: {recording_id: manifest}."""
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.name.endswith('.zarr/.zattrs'):
                d = json.load(tf.extractfile(m))
                if d.get('content_types') == ['2d_brightness_mode']:
                    out[d['recording_id']] = d
    return out


def extract_recordings(tar_path, rids, dest):
    """Yalnız istenen kayıtların üyelerini aç."""
    want = tuple(f'{r}.zarr/' for r in rids)
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf if any(w in m.name for w in want)]
        tf.extractall(dest, members=members, filter='data')


def scan_convert(frames, geometry):
    """(T,R,A) beamspace -> (T,H,W) kartezyen; koordinatlar bir kez hesaplanır."""
    grid = CartesianGrid.from_sector_height(geometry, RENDER_HEIGHT)
    rows, cols, mask = _sector_sample_coordinates(geometry, grid, source_shape=frames.shape[1:3])
    mx = cols.astype(np.float32); my = rows.astype(np.float32)
    out = np.empty((frames.shape[0],) + mask.shape, np.uint8)
    for i, f in enumerate(frames):
        r = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[i] = np.where(mask, r, 0)
    return out, mask


def crop_pad_resize(clip, mask, size=SIZE):
    """Sektör bbox'a kırp -> kareye pad -> resize. redownload_hires.process_clip_hires ile aynı."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = clip[:, y0:y1, x0:x1]
    h, w = c.shape[1:]
    s = max(h, w)
    if h != w:
        sq = np.zeros((c.shape[0], s, s), np.uint8)
        yo, xo = (s - h) // 2, (s - w) // 2
        sq[:, yo:yo + h, xo:xo + w] = c
        c = sq
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in c])


def process_exam(exam_id, tar_path, out_dir, tmp_dir, rows):
    recs = bmode_recordings(tar_path)
    if not recs:
        return 0
    work = os.path.join(tmp_dir, exam_id)
    shutil.rmtree(work, ignore_errors=True)
    extract_recordings(tar_path, recs.keys(), work)
    n = 0
    for rid, att in recs.items():
        outp = os.path.join(out_dir, f'{exam_id}__{rid}.npz')
        if os.path.exists(outp):
            n += 1; continue
        try:
            zp = os.path.join(work, 'exams', exam_id, f'{rid}.zarr')
            sec = att['recording_manifest']['sectors'][0]
            g = sector_geometry_from_mapping(sec['geometry'])
            frames = np.asarray(zarr.open(zp, mode='r')[sec['frames']['zarr_path']])
            if frames.ndim != 3 or frames.shape[0] < 8:
                continue
            car, mask = scan_convert(frames, g)
            clip = crop_pad_resize(car, mask)
            dt = (att.get('median_delta_time') or {}).get('2d_brightness_mode')
            np.savez_compressed(outp, frames=clip, exam_id=exam_id, recording_id=rid,
                                n_orig_frames=frames.shape[0],
                                fps=('' if not dt else round(1.0 / dt, 2)),
                                depth_m=sec['geometry'].get('depth_end_m', ''),
                                beamspace_shape=str(frames.shape))
            rows.append({'exam_id': exam_id, 'recording_id': rid, 'n_frames': clip.shape[0],
                         'fps': ('' if not dt else round(1.0 / dt, 2))})
            n += 1
        except Exception as e:                                    # kayıt atla, muayene devam
            rows.append({'exam_id': exam_id, 'recording_id': rid, 'n_frames': -1, 'fps': f'ERR:{e}'})
    shutil.rmtree(work, ignore_errors=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exams', default='external/render_cohort.txt')
    ap.add_argument('--tars', default='echoxflow/exams')
    ap.add_argument('--out', default=os.path.expanduser('~/echoxflow-render'))
    ap.add_argument('--tmp', default=os.path.expanduser('~/echoxflow-render/_tmp'))
    a = ap.parse_args()
    npz_dir = os.path.join(a.out, 'npz')
    os.makedirs(npz_dir, exist_ok=True); os.makedirs(a.tmp, exist_ok=True)
    exams = [l.strip() for l in open(a.exams) if l.strip()]
    rows, t0 = [], time.time()
    for i, ex in enumerate(exams, 1):
        tar = os.path.join(a.tars, f'{ex}.tar')
        if not os.path.exists(tar):
            print('EKSİK tar', ex, flush=True); continue
        n = process_exam(ex, tar, npz_dir, a.tmp, rows)
        el = time.time() - t0
        print(f'[{i}/{len(exams)}] {ex}  {n} klip  '
              f'{el/60:.1f} dk geçti, tahmini kalan {(el/i)*(len(exams)-i)/60:.0f} dk', flush=True)
        if i % 20 == 0:
            with open(os.path.join(a.out, 'render_manifest.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['exam_id', 'recording_id', 'n_frames', 'fps'])
                w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, 'render_manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exam_id', 'recording_id', 'n_frames', 'fps'])
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r['n_frames'] > 0]
    print(f'BİTTİ: {len(ok)} klip / {len(exams)} muayene, {(time.time()-t0)/60:.0f} dk')


if __name__ == '__main__':
    main()
