#!/usr/bin/env python3
"""A2 (analysis_log Ek 14): ablasyon kontrol noktalarıyla dış kohortta çıkarım. Kilitli tahmin dosyasına dokunmaz.
Çıktı: external/posthoc/ablation_external.csv (exam_id, kosu, p_ham). Eşik/kalibratör kullanılmaz (yalnız AUROC).
Çalıştırma: python3 external/posthoc/infer_ablation.py ~/mimic-echo/runs/rev_abl/*"""
from __future__ import annotations
import collections, csv, glob, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'grading'))
from dataset import AUX_COLS                                      # noqa: E402
from model import DiastolicModel, corn_predict                    # noqa: E402
from run_external import load_clip                                # noqa: E402


def main():
    keep = {r['dicom_id'] for r in csv.DictReader(open(os.path.expanduser('~/echoxflow-render/_views.csv'))) if r['keep'] == '1'}
    files = [p for p in sorted(glob.glob(os.path.expanduser('~/echoxflow-render/npz/*.npz'))) if os.path.basename(p)[:-4] in keep]
    clips = [load_clip(fp) for fp in files]
    rows = []
    for rd in sys.argv[1:]:
        ck = os.path.join(rd, 'best.pt')
        if not os.path.exists(ck) or not os.path.exists(os.path.join(rd, 'binary_results.json')):
            print('atlandı (tamamlanmamış):', rd); continue
        m = DiastolicModel(16, 2, len(AUX_COLS), pretrained=True, dropout=0.25).to('cuda')
        m.load_state_dict(torch.load(ck, weights_only=True)); m.eval()
        s, n = collections.defaultdict(float), collections.Counter()
        for i in range(0, len(clips), 24):
            x = torch.stack([c[0] for c in clips[i:i + 24]]).to('cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                ol, _ = m(x)
            pr = corn_predict(ol.float())[1].float().cpu().numpy()[:, 0]
            for (_, ex), p in zip(clips[i:i + 24], pr):
                s[ex] += float(p); n[ex] += 1
        rows += [[e, os.path.basename(rd.rstrip('/')), f'{s[e] / n[e]:.5f}'] for e in sorted(s)]
        print(os.path.basename(rd), len(s), 'muayene', flush=True)
        del m
    with open(os.path.join(HERE, 'ablation_external.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['exam_id', 'kosu', 'p_ham']); w.writerows(rows)


if __name__ == '__main__':
    main()
