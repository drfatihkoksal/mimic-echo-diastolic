#!/usr/bin/env python3
"""
Faz C — Adım 4-6: eğitim + study-level değerlendirme.

- Per-klip eğitim (CORN ordinal + aux regresyon), study-level toplama ile değerlendirme.
- Dengesizlik: WeightedRandomSampler (ters-frekans, grade bazlı).
- Optimizasyon: AdamW (backbone düşük LR + head yüksek LR), warmup+cosine, AMP(bf16), grad-clip.
- Early-stop: val **quadratic weighted kappa** (QWK) — ordinal için birincil.
- Metrik: QWK + ±1 doğruluk + balanced-acc + confusion; 4-sınıf VE türetilmiş 3-sınıf (G2+G3);
  genel + ecg_sinus alt-kümesi. Test'te en iyi checkpoint + per-study tahmin CSV.

Kullanım:
  python3 train.py --smoke                     # hızlı boru-hattı testi (birkaç adım)
  python3 train.py --epochs 25 --batch 24 --out-dir ~/mimic-echo/runs/c1_panecho
"""
from __future__ import annotations
import argparse, csv, json, math, os, time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, confusion_matrix, f1_score

from dataset import EchoClipDataset, compute_aux_stats, GRADES, AUX_COLS
from model import DiastolicModel, corn_loss, corn_predict, masked_mse


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_train_sampler(ds, n_classes):
    """Ters-frekans (grade bazlı) klip ağırlıkları → dengeli örnekleme."""
    grades = np.array([ds.meta[sid]['grade'] for _, sid in ds.items])
    counts = np.bincount(grades, minlength=n_classes).astype(np.float64)
    cw = counts.sum() / (n_classes * np.maximum(counts, 1))
    w = cw[grades]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)


@torch.no_grad()
def evaluate(model, loader, dev, n_classes, aux_w, max_batches=None):
    """Klip tahminlerini study-level topla, metrik döndür. Döner (metrics, per_study_records, val_loss)."""
    model.eval()
    cum_sum = defaultdict(lambda: np.zeros(n_classes - 1))
    cum_cnt = defaultdict(int)
    s_true, s_conf = {}, {}
    tot_loss, nb = 0.0, 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = b['x'].to(dev, non_blocking=True); y = b['grade'].to(dev)
        aux = b['aux'].to(dev); am = b['aux_mask'].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ol, ap = model(x)
        tot_loss += (corn_loss(ol.float(), y, n_classes) + aux_w * masked_mse(ap.float(), aux, am)).item(); nb += 1
        _, cum = corn_predict(ol.float())
        cum = cum.cpu().numpy()
        for i, sid in enumerate(b['study_id']):
            cum_sum[sid] += cum[i]; cum_cnt[sid] += 1
            s_true[sid] = int(b['grade'][i]); s_conf[sid] = b['confidence'][i]
    # study-level tahmin: kümülatif olasılık ortalaması
    recs = []
    for sid in cum_sum:
        mc = cum_sum[sid] / cum_cnt[sid]
        pred = int((mc > 0.5).sum())
        recs.append({'study_id': sid, 'true': s_true[sid], 'pred': pred,
                     'confidence': s_conf[sid], 'n_clips': cum_cnt[sid],
                     'cum': ';'.join(f'{v:.3f}' for v in mc)})
    return recs, tot_loss / max(nb, 1)


def metrics_from(recs, subset=None):
    r = [x for x in recs if (subset is None or x['confidence'] == subset)]
    if len(r) < 2:
        return None
    t = np.array([x['true'] for x in r]); p = np.array([x['pred'] for x in r])
    def block(tt, pp, labels):
        return {
            'n': int(len(tt)),
            'qwk': round(float(cohen_kappa_score(tt, pp, weights='quadratic', labels=labels)), 4),
            'acc': round(float((tt == pp).mean()), 4),
            'acc_pm1': round(float((np.abs(tt - pp) <= 1).mean()), 4),
            'bacc': round(float(balanced_accuracy_score(tt, pp)), 4),
            'macro_f1': round(float(f1_score(tt, pp, average='macro', labels=labels, zero_division=0)), 4),
        }
    out = {'4class': block(t, p, list(range(4)))}
    # türetilmiş 3-sınıf (Grade3->Grade2 birleşik)
    t3, p3 = np.minimum(t, 2), np.minimum(p, 2)
    out['3class_merged'] = block(t3, p3, list(range(3)))
    out['confusion_4'] = confusion_matrix(t, p, labels=list(range(4))).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.expanduser('~/mimic-echo/runs/c1_panecho'))
    ap.add_argument('--arch', default='panecho', choices=['panecho', 'panecho2d', 'r2plus1d'])
    ap.add_argument('--pretrained', type=int, default=1, help='0=sıfırdan (rastgele init)')
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--n-classes', type=int, default=4)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-4, help='head LR')
    ap.add_argument('--backbone-lr-mult', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--dropout', type=float, default=0.25)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='hızlı boru-hattı testi (2 epoch, az adım)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'
    set_seed(args.seed)
    K = args.n_classes

    aux_stats = compute_aux_stats()
    sf = (args.arch == 'panecho2d')
    tr = EchoClipDataset('train', args.clip_len, train=True, aux_stats=aux_stats, single_frame=sf)
    va = EchoClipDataset('val', args.clip_len, train=False, aux_stats=aux_stats, single_frame=sf)
    print(f"[data] arch={args.arch} pretrained={args.pretrained} single_frame={sf} "
          f"| train klip={len(tr)} val klip={len(va)}", flush=True)
    sampler = make_train_sampler(tr, K)
    tl = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True)

    model = DiastolicModel(args.clip_len, K, len(AUX_COLS), pretrained=bool(args.pretrained),
                           dropout=args.dropout, arch=args.arch).to(dev)
    head_params = list(model.ord_head.parameters()) + list(model.aux_head.parameters())
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * args.backbone_lr_mult},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    steps_ep = 8 if args.smoke else len(tl)
    total = steps_ep * (2 if args.smoke else args.epochs)
    warmup = max(1, int(total * args.warmup_frac))
    def lr_at(step):
        if step < warmup: return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    hist = []; best_qwk = -2.0; best_ep = -1; bad = 0
    gstep = 0; t0 = time.time()
    epochs = 2 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train(); run = 0.0; nb = 0
        for bi, b in enumerate(tl):
            if args.smoke and bi >= steps_ep: break
            x = b['x'].to(dev, non_blocking=True); y = b['grade'].to(dev)
            aux = b['aux'].to(dev); am = b['aux_mask'].to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ol, ap = model(x)
                loss = corn_loss(ol.float(), y, K) + args.aux_weight * masked_mse(ap.float(), aux, am)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); gstep += 1
            run += loss.item(); nb += 1
            if bi % 100 == 0:
                print(f"  ep{ep} step{bi}/{steps_ep} loss={loss.item():.3f} lr={sched.get_last_lr()[1]:.2e}", flush=True)
        recs, vloss = evaluate(model, vl, dev, K, args.aux_weight, max_batches=8 if args.smoke else None)
        M = metrics_from(recs); Meog = metrics_from(recs, 'ecg_sinus')
        qwk = M['4class']['qwk']
        el = (time.time() - t0) / 60
        print(f"[ep{ep}] train_loss={run/max(nb,1):.3f} val_loss={vloss:.3f} | "
              f"QWK={qwk} ±1={M['4class']['acc_pm1']} bacc={M['4class']['bacc']} "
              f"| 3cls-QWK={M['3class_merged']['qwk']} | ecg_sinus-QWK={(Meog or {}).get('4class',{}).get('qwk')} "
              f"| {el:.1f}dk", flush=True)
        hist.append({'epoch': ep, 'train_loss': run/max(nb,1), 'val_loss': vloss, 'val': M, 'val_ecg_sinus': Meog})
        json.dump(hist, open(os.path.join(args.out_dir, 'history.json'), 'w'), indent=1)
        if qwk > best_qwk:
            best_qwk = qwk; best_ep = ep; bad = 0
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep,
                        'aux_stats': aux_stats, 'val_metrics': M}, os.path.join(args.out_dir, 'best.pt'))
            print(f"  ✓ yeni en iyi QWK={qwk} (ep{ep}) → best.pt", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] {args.patience} epoch iyileşme yok (en iyi ep{best_ep} QWK={best_qwk})", flush=True)
                break

    # ---- TEST (en iyi checkpoint) ----
    print(f"\n[test] en iyi checkpoint (ep{best_ep}, val-QWK={best_qwk}) yükleniyor...", flush=True)
    ckpt = torch.load(os.path.join(args.out_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    te = EchoClipDataset('test', args.clip_len, train=False, aux_stats=aux_stats, single_frame=sf)
    tel = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    recs, _ = evaluate(model, tel, dev, K, args.aux_weight, max_batches=8 if args.smoke else None)
    Mt = metrics_from(recs); Mte = metrics_from(recs, 'ecg_sinus')
    with open(os.path.join(args.out_dir, 'test_predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['study_id', 'true', 'pred', 'confidence', 'n_clips', 'cum_probs'])
        for r in recs:
            w.writerow([r['study_id'], GRADES[r['true']], GRADES[r['pred']], r['confidence'], r['n_clips'], r['cum']])
    json.dump({'test_all': Mt, 'test_ecg_sinus': Mte, 'best_epoch': best_ep, 'best_val_qwk': best_qwk},
              open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=1)
    print("=== TEST (genel) ===");        print(json.dumps(Mt, indent=1, ensure_ascii=False))
    print("=== TEST (ecg_sinus) ==="); print(json.dumps(Mte, indent=1, ensure_ascii=False))
    print(f"\nÇıktılar: {args.out_dir}/ (best.pt, history.json, test_metrics.json, test_predictions.csv)")


if __name__ == '__main__':
    main()
