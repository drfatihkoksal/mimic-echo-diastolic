#!/usr/bin/env python3
"""A3 (analysis_log Ek 14): PanEcho'nun önceden eğitilmiş LA çıktıları — LAVol (cm³) ve LASize.
İnce ayar yok. Klip tahminleri çalışma/muayene düzeyinde ortalanır. İç tahminler PhysioNet DUA gereği proje dışında
(~/mimic-echo/runs/rev_panecho_la/), dış tahminler external/posthoc/external_panecho_la.csv.
Çalıştırma (proje kökünden): python3 external/posthoc/panecho_la.py"""
from __future__ import annotations
import collections, csv, glob, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'grading'))
import dataset as ds                                              # noqa: E402
from dataset import EchoClipDataset, compute_aux_stats            # noqa: E402
from run_external import load_clip                                # noqa: E402
SPLITS = os.path.expanduser('~/mimic-echo/splits_binary.csv'); MANIFEST = os.path.expanduser('~/mimic-echo/manifest_study.csv')
OUT_INT = os.path.expanduser('~/mimic-echo/runs/rev_panecho_la'); dev = 'cuda'


def run(model, batches):
    s = collections.defaultdict(lambda: np.zeros(2)); n = collections.Counter()
    with torch.no_grad():
        for x, ids in batches:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(x.to(dev, non_blocking=True))
            vol = o['LAVol'].float().cpu().numpy().reshape(len(ids), -1)[:, 0]
            sz = o['LASize'].float().cpu().numpy()                 # ['Mildly Dilated','Mod|Sev Dilated','Normal']
            assert sz.shape[1] == 3
            for i, k in enumerate(ids):
                s[k] += np.array([vol[i], 1.0 - sz[i, 2]]); n[k] += 1
    return {k: s[k] / n[k] for k in s}, n


def main():
    os.makedirs(OUT_INT, exist_ok=True)
    model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', tasks=['LAVol', 'LASize'], pretrained=True,
                           clip_len=16, trust_repo=True).eval().to(dev)
    ds.GRADES[:] = ['normal', 'elevated']
    aux = compute_aux_stats(SPLITS, MANIFEST)
    for sp in ('val', 'test'):
        d = EchoClipDataset(sp, 16, train=False, aux_stats=aux, splits_csv=SPLITS, manifest_csv=MANIFEST)
        dl = DataLoader(d, batch_size=24, shuffle=False, num_workers=10, pin_memory=True)
        res, n = run(model, ((b['x'], list(b['study_id'])) for b in dl))
        with open(os.path.join(OUT_INT, f'{sp}_la.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['study_id', 'lavol_cm3', 'p_la_dilated', 'n_clips'])
            w.writerows([[k, f'{v[0]:.4f}', f'{v[1]:.5f}', n[k]] for k, v in sorted(res.items())])
        print(f'[iç {sp}] {len(res)} çalışma, {sum(n.values())} klip', flush=True)
    keep = {r['dicom_id'] for r in csv.DictReader(open(os.path.expanduser('~/echoxflow-render/_views.csv'))) if r['keep'] == '1'}
    files = [p for p in sorted(glob.glob(os.path.expanduser('~/echoxflow-render/npz/*.npz'))) if os.path.basename(p)[:-4] in keep]

    def gen():
        buf, ids = [], []
        for fp in files:
            x, ex = load_clip(fp); buf.append(x); ids.append(ex)
            if len(buf) == 24:
                yield torch.stack(buf), ids; buf, ids = [], []
        if buf:
            yield torch.stack(buf), ids
    res, n = run(model, gen())
    with open(os.path.join(HERE, 'external_panecho_la.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['exam_id', 'lavol_cm3', 'p_la_dilated', 'n_clips'])
        w.writerows([[k, f'{v[0]:.4f}', f'{v[1]:.5f}', n[k]] for k, v in sorted(res.items())])
    print(f'[dış] {len(res)} muayene, {len(files)} klip', flush=True)


if __name__ == '__main__':
    main()
