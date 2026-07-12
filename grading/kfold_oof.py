#!/usr/bin/env python3
"""
Faz C — Prognostik güçlendirme: 5-kat OUT-OF-FOLD tahmin.

Amaç: modelin YANSIZ tahmini her study için (o study'nin hastası eğitimde YOKKEN) → tüm 3065
study'ye tahmin → mortalite survival'ı 300 ölümle güçlendir (test-seti 37 ölümle az-güçlüydü).

Tasarım: subject-bazlı StratifiedGroupKFold(5). Her kat: diğer 4 kat eğit (sabit epoch,
early-stop yok — CV'de model seçimi yok), tutulan katı tahmin et. Kümülatif olasılıkları
per-study topla. Çıktı: ~/mimic-echo/runs/oof_predictions.csv (study_id,true,pred,cum_probs,split_orig).

Kullanım (arka plan): nohup python3 kfold_oof.py --folds 5 --epochs 8 > ~/mimic-echo/oof.log 2>&1 &
"""
from __future__ import annotations
import argparse, csv, os, math, time, tempfile
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

from dataset import EchoClipDataset, compute_aux_stats, GRADES, AUX_COLS, DEFAULT_SPLITS, DEFAULT_MANIFEST
from model import DiastolicModel, corn_loss, corn_predict, masked_mse


def load_rows():
    rows = []
    with open(DEFAULT_SPLITS, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)  # study_id, subject_id, grade_2025, confidence, split, n_clips
    return rows


def write_temp_split(rows, oof_studies, val_studies, path):
    """oof→'test'; val→'val'; gerisi→'train'. Diğer kolonlar korunur."""
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows:
            rr = dict(r)
            rr['split'] = 'test' if r['study_id'] in oof_studies else ('val' if r['study_id'] in val_studies else 'train')
            w.writerow(rr)


@torch.no_grad()
def predict(model, ds, dev, batch, workers):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    cs = defaultdict(lambda: np.zeros(3)); cc = defaultdict(int); tg = {}
    for b in dl:
        x = b['x'].to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ol, _ = model(x)
        _, cum = corn_predict(ol.float()); cum = cum.cpu().numpy()
        for i, sid in enumerate(b['study_id']):
            cs[sid] += cum[i]; cc[sid] += 1; tg[sid] = int(b['grade'][i])
    out = {}
    for sid in cs:
        mc = cs[sid] / cc[sid]
        out[sid] = dict(true=tg[sid], pred=int((mc > 0.5).sum()), cum=mc)
    return out


def val_qwk(model, ds, dev, workers):
    from sklearn.metrics import cohen_kappa_score
    p = predict(model, ds, dev, 48, workers)
    t = np.array([v['true'] for v in p.values()]); pr = np.array([v['pred'] for v in p.values()])
    return float(cohen_kappa_score(t, pr, weights='quadratic', labels=[0, 1, 2, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max-epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    dev = 'cuda'; K = 4
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    aux_stats = compute_aux_stats()
    ckpt = os.path.join(tempfile.gettempdir(), 'oof_fold_best.pt')

    rows = load_rows()
    orig_split = {r['study_id']: r['split'] for r in rows}
    sids = [r['study_id'] for r in rows]
    y = np.array([GRADES.index(r['grade_2025']) for r in rows])
    groups = np.array([r['subject_id'] for r in rows])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = {}
    outdir = os.path.expanduser('~/mimic-echo/runs'); os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    for fold, (tr_idx, oof_idx) in enumerate(sgkf.split(sids, y, groups)):
        oof_studies = set(sids[i] for i in oof_idx)
        # RESUME: bu fold zaten diskte varsa yükle, atla (çökme-dayanıklı)
        fold_csv = os.path.join(outdir, f'oof_fold{fold}.csv')
        if os.path.exists(fold_csv):
            with open(fold_csv, newline='') as f:
                for r in csv.DictReader(f):
                    oof[r['study_id']] = dict(true=GRADES.index(r['true']), pred=GRADES.index(r['pred']),
                                              cum=np.array([float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])]))
            print(f"[fold{fold}] diskten yüklendi (atlandı): {fold_csv}", flush=True)
            continue
        # NESTED: kalan 4 kattan subject-bazlı iç-validasyon ayır (OOF katına DOKUNMADAN)
        sub_sids = [sids[i] for i in tr_idx]; sub_y = y[tr_idx]; sub_g = groups[tr_idx]
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed + fold)
        _, val_local = next(inner.split(sub_sids, sub_y, sub_g))
        val_studies = set(sub_sids[j] for j in val_local)
        tmp = os.path.join(tempfile.gettempdir(), f'oof_split_{fold}.csv')
        write_temp_split(rows, oof_studies, val_studies, tmp)

        tr = EchoClipDataset('train', 16, train=True, aux_stats=aux_stats, splits_csv=tmp)
        va = EchoClipDataset('val', 16, train=False, aux_stats=aux_stats, splits_csv=tmp)
        g = np.array([tr.meta[s]['grade'] for _, s in tr.items])
        cnt = np.bincount(g, minlength=K); w = (cnt.sum() / (K * np.maximum(cnt, 1)))[g]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        tl = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
        model = DiastolicModel(16, K, len(AUX_COLS), pretrained=True, arch='panecho').to(dev)
        head = list(model.ord_head.parameters()) + list(model.aux_head.parameters())
        opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
                                 {'params': head, 'lr': args.lr}], weight_decay=1e-4)
        total = len(tl) * args.max_epochs; warm = max(1, int(0.05 * total))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else
                                                0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total - warm), 1.0))))

        # ── EPOCH-BAZLI SÜRDÜRME ────────────────────────────────────────────────
        # GPU bu makinede uzun yük altında düşebiliyor (GSP kilitlenmesi → kart PCIe'den
        # kayboluyor). Fold ~50dk sürdüğü için fold-bazlı checkpoint yetmiyordu: her çöküş
        # 50dk'yı çöpe atıyordu. Artık HER EPOCH sonunda tam durum (model+opt+sched+en iyi)
        # diske yazılıyor → çöküşte en fazla 1 epoch (~4dk) kaybediliyor.
        state_f = os.path.join(outdir, f'oof_state_fold{fold}.pt')
        start_ep, best, bad, best_sd = 0, -2.0, 0, None
        if os.path.exists(state_f):
            st = torch.load(state_f, map_location=dev, weights_only=False)
            model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
            sch.load_state_dict(st['sch'])
            start_ep, best, bad, best_sd = st['next_ep'], st['best'], st['bad'], st['best_sd']
            print(f"  [fold{fold}] SÜRDÜRÜLÜYOR: ep{start_ep}'ten devam (en iyi={best:.3f})", flush=True)

        for ep in range(start_ep, args.max_epochs):
            model.train()
            for b in tl:
                x = b['x'].to(dev, non_blocking=True); yb = b['grade'].to(dev)
                aux = b['aux'].to(dev); am = b['aux_mask'].to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ol, apr = model(x)
                    loss = corn_loss(ol.float(), yb, K) + 0.3 * masked_mse(apr.float(), aux, am)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sch.step()
            q = val_qwk(model, va, dev, args.workers)          # iç-val QWK (early-stop metriği)
            if q > best:
                best = q; bad = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            # her epoch: tam durumu ATOMİK yaz (yarım dosya bırakma)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(),
                        'next_ep': ep + 1, 'best': best, 'bad': bad, 'best_sd': best_sd},
                       state_f + '.tmp')
            os.replace(state_f + '.tmp', state_f)
            print(f"  [fold{fold}] ep{ep} val-QWK={q:.3f}{' *' if bad==0 else ''} ({(time.time()-t0)/60:.0f}dk)", flush=True)
            if bad >= args.patience:
                print(f"  [fold{fold}] early-stop (en iyi val-QWK={best:.3f})", flush=True); break
        # en iyi ağırlıklarla OOF tahmin
        model.load_state_dict(best_sd)
        te = EchoClipDataset('test', 16, train=False, aux_stats=aux_stats, splits_csv=tmp)
        p = predict(model, te, dev, 48, args.workers)
        oof.update(p)
        # per-fold CSV (çökme-dayanıklı + interim analiz): fold biter bitmez diske yaz
        with open(fold_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['study_id', 'true', 'pred', 'p_ge1', 'p_ge2', 'p_ge3'])
            for sid, v in p.items():
                w.writerow([sid, GRADES[v['true']], GRADES[v['pred']],
                            f"{v['cum'][0]:.4f}", f"{v['cum'][1]:.4f}", f"{v['cum'][2]:.4f}"])
        print(f"[fold{fold}] tamam: {len(oof_studies)} OOF study → {fold_csv} | toplam={len(oof)} "
              f"| best-val-QWK={best:.3f} | {(time.time()-t0)/60:.0f}dk", flush=True)
        # fold bitti → epoch-durumu artık gereksiz (her biri ~0.5GB), sil
        if os.path.exists(state_f):
            os.remove(state_f)
        del model, tl, tr, va; torch.cuda.empty_cache()

    out = os.path.expanduser('~/mimic-echo/runs/oof_predictions.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['study_id', 'true', 'pred', 'p_ge1', 'p_ge2', 'p_ge3', 'split_orig'])
        for sid, v in oof.items():
            w.writerow([sid, GRADES[v['true']], GRADES[v['pred']], f"{v['cum'][0]:.4f}",
                        f"{v['cum'][1]:.4f}", f"{v['cum'][2]:.4f}", orig_split.get(sid, '')])
    # hızlı OOF metrik
    from sklearn.metrics import cohen_kappa_score
    t = np.array([v['true'] for v in oof.values()]); pr = np.array([v['pred'] for v in oof.values()])
    print(f"\nOOF BİTTİ: {len(oof)} study | QWK(4)={cohen_kappa_score(t,pr,weights='quadratic'):.3f} "
          f"| 3cls QWK={cohen_kappa_score(np.minimum(t,2),np.minimum(pr,2),weights='quadratic'):.3f}")
    print(f"Kaydedildi: {out} | süre {(time.time()-t0)/60:.0f}dk")


if __name__ == '__main__':
    main()
