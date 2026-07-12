#!/usr/bin/env python3
"""
C.2 MIL modelinin test tahminlerini CSV'ye dök — CPU'da, GPU'yu meşgul etmeden.

Neden: train_mil.py yalnız özet metrik (test_metrics.json) yazıyordu; per-study tahmin yoktu.
Bu yüzden MIL için güven aralığı / ikili metrik / mortalite analizi yapılamıyordu ve paper'da
MIL'in AUROC'u C.1'in güven aralığıyla raporlanmıştı. Embedding'ler önbellekte (emb_test.npz)
ve MIL head küçük olduğu için tahminler CPU'da saniyeler içinde yeniden üretilebilir.

Çıktı: runs/c2_mil/<pool>/test_predictions.csv — C.1 ile AYNI format
       (study_id,true,pred,confidence,n_clips,cum_probs) → binary_eval.py ve
       mortality_analysis.py doğrudan tüketebilir.

Kullanım:
  python3 mil_predict.py --pool attention --split test
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import compute_aux_stats, AUX_COLS, GRADES
from model import corn_predict
from train_mil import BagDS, AttnMIL, collate, load_aux, EMB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='attention', choices=['attention', 'mean'])
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    outdir = os.path.join(EMB, f'mil_{args.pool}')
    ckpt = os.path.join(outdir, 'best.pt')
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint yok: {ckpt}")

    aux_mean, aux_std = compute_aux_stats()
    ds = BagDS(args.split, load_aux(), aux_mean, aux_std)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    model = AttnMIL(n_aux=len(AUX_COLS), pool=args.pool).to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
    model.eval()

    out = os.path.join(outdir, f'{args.split}_predictions.csv')
    n = 0
    with torch.no_grad(), open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['study_id', 'true', 'pred', 'confidence', 'n_clips', 'cum_probs'])
        for x, mask, grade, _aux, _am, sids, confs in dl:
            ol, _ = model(x.to(args.device), mask.to(args.device))
            pred, cum = corn_predict(ol.float())
            cum = cum.cpu().numpy(); pred = pred.cpu().numpy()
            n_clips = mask.sum(1).cpu().numpy()
            for i, sid in enumerate(sids):
                w.writerow([sid, GRADES[int(grade[i])], GRADES[int(pred[i])], confs[i], int(n_clips[i]),
                            ';'.join(f"{v:.4f}" for v in cum[i])])
                n += 1
    print(f"{n} study → {out}")


if __name__ == '__main__':
    main()
