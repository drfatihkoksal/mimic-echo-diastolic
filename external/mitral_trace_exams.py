#!/usr/bin/env python3
"""
Ön-tanımlı mitral duyarlılık analizi (analysis_plan §5.4) için muayene listesi.

Ölçüt: sonografın mitral CW izi, MR izi ya da PISA izi çizdiği muayene. Mitral zarf PW izinde
çizilmişse (giriş akımı VTI'ı) ölçüte GİRMEZ; yalnız CW izindeki mitral trace sayılır.
Yalnız tar başlıklarındaki .zattrs okunur. Çıktı: external/mitral_trace_exams.json
"""
import glob, json, os, tarfile
from multiprocessing import Pool


def one(path):
    ex = os.path.basename(path)[:-4]; hits = set()
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.name.endswith('.zarr/.zattrs'):
                continue
            man = json.load(tf.extractfile(m)).get('recording_manifest', {}) or {}
            for t in man.get('tracks') or []:
                for a in t.get('spectral_annotations') or []:
                    lab = a.get('label') or ''
                    if 'PISA' in lab or 'MR Trace' in lab:
                        hits.add(lab)
                    elif 'Mitral Valve' in lab and 'Trace' in lab and t.get('semantic_id') == 'continuous_wave':
                        hits.add(lab + ' [CW]')
    return ex, sorted(hits)


if __name__ == '__main__':
    tars = sorted(glob.glob('echoxflow/exams/*.tar'))
    with Pool(8) as p:
        res = dict(p.map(one, tars))
    out = {k: v for k, v in res.items() if v}
    json.dump({'olcut': 'mitral CW izi, MR izi veya PISA izi (analysis_plan §5.4)', 'muayeneler': out},
              open('external/mitral_trace_exams.json', 'w'), indent=1, ensure_ascii=False)
    print(len(out), 'muayene')
