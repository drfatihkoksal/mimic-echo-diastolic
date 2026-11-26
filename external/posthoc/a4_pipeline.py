#!/usr/bin/env python3
"""A4 (analysis_log Ek 16): analiz kohortunun dışındaki çalışmalarda kilitli modelin çıktısı.
Aşamalar (sırayla):  prep112 -> [grading/view_classify.py] -> hires -> infer
Bütün ara dosyalar ve çalışma düzeyi tahminler PhysioNet DUA gereği proje dışında: ~/mimic-echo/rev_a4/
Çalıştırma (proje kökünden):  python3 external/posthoc/a4_pipeline.py prep112|hires|infer [--limit N]"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'grading'))
from shrink_pipeline import classify, process_clip            # noqa: E402

H = os.path.expanduser('~/mimic-echo/')
ROOT = os.environ.get('MIMIC_ECHO_DICOM_ROOT', os.path.expanduser('~/mimic-echo/dicom'))
OUT = os.path.join(H, 'rev_a4')
TARGETS = {'A4C': 'A4C', 'A2C': 'A2C', 'Parasternal_Long': 'PLAX'}


def target_studies():
    man = {r['study_id'] for r in csv.DictReader(open(H + 'manifest_study.csv'))}
    return {r['study_id'] for r in csv.DictReader(open(H + 'diastolic_labels.csv')) if r['study_id'] not in man}


def job112(a):
    path, sid, out_dir = a
    did = os.path.basename(path)[:-4]; outp = os.path.join(out_dir, did + '.npz')
    if os.path.exists(outp):
        return 'exists'
    try:
        hdr = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return 'unreadable'
    cat, bbox, meta = classify(hdr)
    if cat != 'bmode_cine':
        return cat
    try:
        clip = process_clip(pydicom.dcmread(path), bbox, meta['photometric'], 112, 32)
    except Exception:
        return 'decode_fail'
    np.savez_compressed(outp, frames=clip, study_id=sid, n_orig_frames=meta['n_frames'], cine_rate=str(meta['cine_rate']))
    return 'bmode_cine_kept'


def jobhires(a):
    path, sid, view, prob, out_dir = a
    from redownload_hires import process_clip_hires
    did = os.path.basename(path)[:-4]; outp = os.path.join(out_dir, did + '.npz')
    if os.path.exists(outp):
        return 'exists'
    try:
        ds = pydicom.dcmread(path); cat, bbox, meta = classify(ds)
        clip = process_clip_hires(ds, bbox, 224)
        if clip is None or clip.shape[0] < 4:
            return 'empty'
        np.savez_compressed(outp, frames=clip, view=TARGETS[view], view_prob=str(prob), study_id=sid,
                            n_orig_frames=meta['n_frames'], cine_rate=str(meta['cine_rate']))
        return 'ok'
    except Exception as e:
        return f'err:{type(e).__name__}'


def pool(fn, todo, workers, tag):
    stat = collections.Counter(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in todo]
        for f in as_completed(futs):
            stat[f.result()] += 1; done += 1
            if done % 10000 == 0:
                print(f'  [{tag}] {done}/{len(todo)} {dict(stat)}', flush=True)
    print(f'[{tag}] BİTTİ {dict(stat)}', flush=True)
    json.dump(dict(stat), open(os.path.join(OUT, f'{tag}_stat.json'), 'w'), indent=1)


def infer():
    import torch
    from dataset import _sample_indices, IMAGENET_MEAN, IMAGENET_STD, AUX_COLS
    from model import DiastolicModel, corn_predict
    from train_binary import apply_platt
    run = os.path.join(H, 'runs/b2_binary'); L = json.load(open(os.path.join(run, 'binary_results.json')))
    ab = (L['kalibrasyon']['a'], L['kalibrasyon']['b'])
    m = DiastolicModel(16, 2, len(AUX_COLS), pretrained=True, dropout=0.25).to('cuda')
    m.load_state_dict(torch.load(os.path.join(run, 'best.pt'), weights_only=True)); m.eval()
    files = sorted(glob.glob(os.path.join(OUT, 'npz_hires', '*.npz')))
    s, n = collections.defaultdict(float), collections.Counter(); buf, ids = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ol, _ = m(torch.stack(buf).to('cuda'))
        for sid, p in zip(ids, corn_predict(ol.float())[1].float().cpu().numpy()[:, 0]):
            s[sid] += float(p); n[sid] += 1
        buf.clear(); ids.clear()
    for i, fp in enumerate(files, 1):
        z = np.load(fp, allow_pickle=True); fr = z['frames']
        clip = fr[_sample_indices(fr.shape[0], 16, False)].astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)
        buf.append((x - IMAGENET_MEAN) / IMAGENET_STD); ids.append(str(z['study_id']))
        if len(buf) >= 24:
            flush()
        if i % 5000 == 0:
            print(f'  [infer] {i}/{len(files)}', flush=True)
    flush()
    with open(os.path.join(OUT, 'predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['study_id', 'p_ham', 'p_kalibre', 'n_clips'])
        for sid in sorted(s):
            raw = s[sid] / n[sid]
            w.writerow([sid, f'{raw:.5f}', f'{float(apply_platt(np.array([raw]), ab)[0]):.5f}', n[sid]])
    print(f'[infer] {len(s)} çalışma, {len(files)} klip', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prep112', 'hires', 'infer'])
    ap.add_argument('--workers', type=int, default=14); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.stage == 'infer':
        return infer()
    tgt = target_studies()
    rec = [(os.path.join(ROOT, r['dicom_filepath']), r['study_id'])
           for r in csv.DictReader(open(os.path.join(ROOT, 'echo-record-list.csv'))) if r['study_id'] in tgt]
    if a.stage == 'prep112':
        d = os.path.join(OUT, 'npz112'); os.makedirs(d, exist_ok=True)
        todo = [(p, s, d) for p, s in rec]
        if a.limit:
            todo = todo[:a.limit]
        print(f'[prep112] hedef çalışma {len(tgt)}, DICOM {len(todo)}', flush=True)
        pool(job112, todo, a.workers, 'prep112')
    else:
        d = os.path.join(OUT, 'npz_hires'); os.makedirs(d, exist_ok=True)
        path = {os.path.basename(p)[:-4]: (p, s) for p, s in rec}
        todo = [(path[r['dicom_id']][0], r['study_id'], r['pred_view'], r['prob'], d)
                for r in csv.DictReader(open(os.path.join(OUT, '_views.csv')))
                if r['pred_view'] in TARGETS and r['dicom_id'] in path]
        if a.limit:
            todo = todo[:a.limit]
        print(f'[hires] hedef görünümlü klip {len(todo)}', flush=True)
        pool(jobhires, todo, a.workers, 'hires')


if __name__ == '__main__':
    main()
