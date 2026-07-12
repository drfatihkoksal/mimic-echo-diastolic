#!/usr/bin/env python3
"""
Faz C — Ensemble (soft-voting) performansı, KİLİTLİ test üzerinde (yansız).

Tasarım (OOF'tan farkı — bias yok): orijinal 438 TEST seti HİÇBİR fold'a girmez (kilitli).
Kalan pool (train+val = 2627 study) subject-bazlı 5-kata bölünür; her kat için model, 4 pool-katında
eğitilir, 5. pool-katı iç-validasyon (early-stop) olur, KİLİTLİ test tahmin edilir. 5 modelin
kümülatif olasılıkları test'te ORTALANIR (soft-voting) → ensemble; tek-model ortalamasıyla kıyaslanır.
Bonus: 5 model diske saklanır → dağıtılabilir ensemble (dış/gelecek veri, robustluk).

Kullanım (arka plan, OOF bitince): nohup python3 ensemble_eval.py > ~/mimic-echo/ensemble.log 2>&1 &
"""
from __future__ import annotations
import argparse, csv, os, math, time, tempfile
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score

from dataset import EchoClipDataset, compute_aux_stats, GRADES, AUX_COLS, DEFAULT_SPLITS
from model import DiastolicModel, corn_loss, corn_predict, masked_mse
from kfold_oof import load_rows, predict

OUT = os.path.expanduser('~/mimic-echo/runs/ensemble'); os.makedirs(OUT, exist_ok=True)


def write_split(rows, test_studies, val_studies, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows:
            rr = dict(r)
            rr['split'] = 'test' if r['study_id'] in test_studies else ('val' if r['study_id'] in val_studies else 'train')
            w.writerow(rr)


def val_qwk(model, ds, dev, workers):
    p = predict(model, ds, dev, 48, workers)
    t = np.array([v['true'] for v in p.values()]); pr = np.array([v['pred'] for v in p.values()])
    return float(cohen_kappa_score(t, pr, weights='quadratic', labels=[0, 1, 2, 3]))


def metrics(true, pred, cum):
    """true,pred: (N,) ordinal; cum: (N,3) [P≥1,P≥2,P≥3]."""
    t = np.array(true); p = np.array(pred); c = np.array(cum)
    out = {'n': len(t),
           'qwk4': round(float(cohen_kappa_score(t, p, weights='quadratic', labels=[0, 1, 2, 3])), 4),
           'qwk3': round(float(cohen_kappa_score(np.minimum(t, 2), np.minimum(p, 2), weights='quadratic', labels=[0, 1, 2])), 4),
           'acc_pm1': round(float((np.abs(t - p) <= 1).mean()), 4),
           'bacc': round(float(balanced_accuracy_score(t, p)), 4)}
    y1 = (t >= 1).astype(int); y2 = (t >= 2).astype(int)
    if len(np.unique(y1)) == 2: out['AUROC_anyDD'] = round(float(roc_auc_score(y1, c[:, 0])), 3)
    if len(np.unique(y2)) == 2: out['AUROC_advDD'] = round(float(roc_auc_score(y2, c[:, 1])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max-epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=4)
    # val-QWK epoch'lar arası ±0.05 zıplıyor — bu, eğitimin toplam kazancıyla aynı büyüklükte.
    # Yalnız patience ile durdurunca ep0 şansa yüksek çıkan fold'lar 5. epoch'ta kesiliyor ve
    # "en iyi" ağırlık olarak 1 epoch'luk model kalıyordu (OOF koşusunda fold 2/3/4'te oldu:
    # OOF-QWK 0.26–0.32, düzgün eğitilen fold'larda 0.41–0.43). Taban epoch bunu engelliyor.
    # --min-epochs 0 eski davranışa döner.
    ap.add_argument('--min-epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    dev = 'cuda'; K = 4
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    aux_stats = compute_aux_stats()

    rows = load_rows()
    test_studies = set(r['study_id'] for r in rows if r['split'] == 'test')       # KİLİTLİ
    pool = [r for r in rows if r['split'] != 'test']
    psid = [r['study_id'] for r in pool]
    py = np.array([GRADES.index(r['grade_2025']) for r in pool])
    pg = np.array([r['subject_id'] for r in pool])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f"kilitli test={len(test_studies)} | pool={len(pool)} study | {args.folds}-kat ensemble", flush=True)

    test_true = None
    fold_cum = []   # her modelin test kümülatif olasılıkları (study sırası sabit)
    single = []     # tek-model test metrikleri
    t0 = time.time()
    for fold, (tr_local, val_local) in enumerate(sgkf.split(psid, py, pg)):
        val_studies = set(psid[j] for j in val_local)
        tmp = os.path.join(tempfile.gettempdir(), f'ens_split_{fold}.csv')
        write_split(rows, test_studies, val_studies, tmp)
        # RESUME (fold): bir fold ANCAK test tahminleri diske yazıldıysa bitmiş sayılır.
        # model_fold*.pt'nin varlığına bakmak yanlıştı — o dosya eğitim sırasında her en-iyi
        # epoch'ta yazılıyor, dolayısıyla fold ortasındaki bir çöküşten sonra yarım eğitilmiş
        # model "bitmiş fold" sanılıp atlanıyor ve ensemble sessizce zayıf kuruluyordu.
        fold_csv = os.path.join(OUT, f'ens_fold{fold}_test.csv')
        if os.path.exists(fold_csv):
            with open(fold_csv, newline='') as f:
                rr = sorted(csv.DictReader(f), key=lambda r: r['study_id'])
            if test_true is None:
                test_true = [GRADES.index(r['true']) for r in rr]
            cum = np.array([[float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])] for r in rr])
            fold_cum.append(cum)
            single.append(metrics(test_true, [GRADES.index(r['pred']) for r in rr], cum))
            print(f"[fold{fold}] diskten yüklendi (atlandı) | test QWK4={single[-1]['qwk4']}", flush=True)
            continue
        tr = EchoClipDataset('train', 16, train=True, aux_stats=aux_stats, splits_csv=tmp)
        va = EchoClipDataset('val', 16, train=False, aux_stats=aux_stats, splits_csv=tmp)
        te = EchoClipDataset('test', 16, train=False, aux_stats=aux_stats, splits_csv=tmp)
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
        ckpt = os.path.join(OUT, f'model_fold{fold}.pt')

        # ── EPOCH-BAZLI SÜRDÜRME (aynı desen: kfold_oof.py) ──────────────────────
        # GPU bu makinede uzun yük altında düşüyor ve fold ~50dk sürüyor; fold-bazlı
        # checkpoint yetmez. Her epoch sonunda tam durum ATOMİK yazılır → çöküşte en
        # fazla 1 epoch (~4dk) kaybedilir, yeniden başlayınca o epoch'tan devam edilir.
        state_f = os.path.join(OUT, f'ens_state_fold{fold}.pt')
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
            q = val_qwk(model, va, dev, args.workers)
            if q > best:
                best = q; bad = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(),
                        'next_ep': ep + 1, 'best': best, 'bad': bad, 'best_sd': best_sd},
                       state_f + '.tmp')
            os.replace(state_f + '.tmp', state_f)
            print(f"  [fold{fold}] ep{ep} val-QWK={q:.3f}{' *' if bad==0 else ''} ({(time.time()-t0)/60:.0f}dk)", flush=True)
            if bad >= args.patience and (ep + 1) >= args.min_epochs:
                print(f"  [fold{fold}] early-stop (ep{ep}, en iyi val-QWK={best:.3f})", flush=True); break

        model.load_state_dict(best_sd)
        p = predict(model, te, dev, 48, args.workers)          # KİLİTLİ test (yansız)
        sids = sorted(p.keys())
        if test_true is None:
            test_true = [p[s]['true'] for s in sids]
        cum = np.array([p[s]['cum'] for s in sids]); fold_cum.append(cum)
        single.append(metrics(test_true, [p[s]['pred'] for s in sids], cum))
        # fold BİTTİ → dağıtılabilir model + test tahminleri. fold_csv tamamlanma işaretidir,
        # bu yüzden en son ve atomik yazılır: varlığı "bu fold gerçekten bitti" demektir.
        torch.save(best_sd, ckpt + '.tmp'); os.replace(ckpt + '.tmp', ckpt)
        with open(fold_csv + '.tmp', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['study_id', 'true', 'pred', 'p_ge1', 'p_ge2', 'p_ge3'])
            for s in sids:
                v = p[s]
                w.writerow([s, GRADES[v['true']], GRADES[v['pred']],
                            f"{v['cum'][0]:.4f}", f"{v['cum'][1]:.4f}", f"{v['cum'][2]:.4f}"])
        os.replace(fold_csv + '.tmp', fold_csv)
        print(f"[fold{fold}] tamam | best-val-QWK={best:.3f} | test QWK4={single[-1]['qwk4']} "
              f"advDD-AUC={single[-1].get('AUROC_advDD')} | {(time.time()-t0)/60:.0f}dk", flush=True)
        # fold bitti → epoch-durumu artık gereksiz (~0.5GB), sil
        if os.path.exists(state_f):
            os.remove(state_f)
        del model, tl, tr, va, te; torch.cuda.empty_cache()

    # ENSEMBLE: kümülatif olasılıkları ortala (soft-voting)
    ens_cum = np.mean(fold_cum, axis=0)
    ens_pred = (ens_cum > 0.5).sum(axis=1)
    ens = metrics(test_true, ens_pred, ens_cum)
    single_mean = {k: round(float(np.mean([s[k] for s in single])), 4) for k in single[0] if k != 'n'}
    import json
    json.dump({'single_folds': single, 'single_mean': single_mean, 'ensemble': ens},
              open(os.path.join(OUT, 'ensemble_metrics.json'), 'w'), indent=1, ensure_ascii=False)
    print("\n=== ENSEMBLE (soft-voting, kilitli test) ===")
    print(f"tek-model ORTALAMA: QWK4={single_mean['qwk4']} QWK3={single_mean['qwk3']} "
          f"advDD-AUC={single_mean.get('AUROC_advDD')}")
    print(f"ENSEMBLE (5-model): QWK4={ens['qwk4']} QWK3={ens['qwk3']} ±1={ens['acc_pm1']} "
          f"bacc={ens['bacc']} anyDD-AUC={ens.get('AUROC_anyDD')} advDD-AUC={ens.get('AUROC_advDD')}")
    print(f"kazanç: QWK4 {ens['qwk4']-single_mean['qwk4']:+.3f} | 5 model → {OUT}/model_fold*.pt")


if __name__ == '__main__':
    main()
