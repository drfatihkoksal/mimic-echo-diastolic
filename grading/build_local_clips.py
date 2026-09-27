#!/usr/bin/env python3
"""
Ek görünüm kliplerini YEREL DICOM'lardan üretir (indirme yok).

Faz A yalnızca A4C/A2C/PLAX kliplerini 224x224 olarak yeniden indirmişti; A3C, A5C ve
PSAX klipleri sınıflandırılmış ama üretilmemişti. Tam veri seti artık yerelde olduğu
için bunlar indirme olmadan üretilebiliyor: kohort içinde 32.511 klip.

İşleme, mevcut kliplerle BİREBİR aynı (redownload_hires.process_clip_hires):
tüm kareler, Y-luma, ultrason sektörüne kırp, kareye pad, 224x224.
Çıktı ayrı dizine yazılır; mevcut 34.816 klip dokunulmadan kalır.
"""
from __future__ import annotations
import argparse, csv, io, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_pipeline import classify                  # noqa: E402
from redownload_hires import process_clip_hires       # noqa: E402

H = os.path.expanduser('~/mimic-echo/')
ROOT = os.environ.get('MIMIC_ECHO_DICOM_ROOT', os.path.expanduser('~/mimic-echo/dicom'))
EXTRA = {'A3C', 'A5C', 'Parasternal_Short'}
ABBR = {'Parasternal_Short': 'PSAX'}


def job(args):
    did, sid, view, prob, path, out_dir, size = args
    outp = os.path.join(out_dir, f'{did}.npz')
    if os.path.exists(outp):
        return 'exists'
    try:
        ds = pydicom.dcmread(path)
        cat, bbox, meta = classify(ds)
        if cat != 'bmode_cine':
            return 'not_cine'
        clip = process_clip_hires(ds, bbox, size)
        if clip is None or clip.shape[0] < 4:
            return 'empty'
        np.savez_compressed(outp, frames=clip, view=ABBR.get(view, view), view_prob=str(prob),
                            study_id=sid, n_orig_frames=meta['n_frames'],
                            cine_rate=str(meta['cine_rate']), rows=meta['rows'],
                            cols=meta['cols'], bbox=str(bbox), photometric=meta['photometric'])
        return 'ok'
    except Exception as e:
        return f'err:{type(e).__name__}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(H, 'npz_views'))
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    kept = {r['dicom_id'] for r in csv.DictReader(open(H + 'redownload_kept.csv'))}
    splits = {r['study_id'] for r in csv.DictReader(open(H + 'splits_binary.csv'))}
    rec = {os.path.basename(r['dicom_filepath'])[:-4]: r['dicom_filepath']
           for r in csv.DictReader(open(os.path.join(ROOT, 'echo-record-list.csv')))}
    todo = []
    for r in csv.DictReader(open(H + 'npz/_views.csv')):
        if (r['pred_view'] in EXTRA and r['study_id'] in splits
                and r['dicom_id'] not in kept and r['dicom_id'] in rec):
            todo.append((r['dicom_id'], r['study_id'], r['pred_view'], r['prob'],
                         os.path.join(ROOT, rec[r['dicom_id']]), a.out, a.size))
    if a.limit:
        todo = todo[:a.limit]
    print(f'işlenecek: {len(todo)} klip -> {a.out}', flush=True)

    stat, done = {}, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, t) for t in todo]
        for f in as_completed(futs):
            s = f.result(); stat[s] = stat.get(s, 0) + 1; done += 1
            if done % 2000 == 0:
                print(f'  {done}/{len(todo)}  {stat}', flush=True)
    print('BİTTİ', stat, flush=True)


if __name__ == '__main__':
    main()
