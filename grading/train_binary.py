#!/usr/bin/env python3
"""
İkili eğitim: yüksek sol atriyal basınç var/yok (2025 ASE `lap`).

Pivot gerekçesi: JASE hakemleri ve editörü 4-sınıf derecelendirmenin mütevazı
kaldığını, modelin asıl olarak klinik anlamlı disfonksiyon için bir tarama/triyaj
aracı olduğunu belirtti. Sıralı görev tamamen bırakıldı.

Tasarım notları:
- Kohort ve hasta düzeyi bölmeler ÖNCEKİYLE BİREBİR AYNI (3065 çalışma), yalnızca
  etiket ikili. Bölme değişmediği için mevcut encoder val/test hastalarını hiç
  görmedi; sıfırdan ince ayar da aynı ölçüde geçerli.
- Model cerrahisi yok: CORN kaybı K=2'de tam olarak ikili lojistik kayba indirgenir
  (tek mantık), corn_predict'in kümülatif çıktısı doğrudan P(yüksek LAP).
- Model seçimi val AUROC ile (ikili görevde kappa kötü bir seçim ölçütüdür).
- EŞİK YALNIZCA VAL'DE seçilir ve kilitlenir; test ve dış kohorta değiştirilmeden
  uygulanır (JASE Hakem 1 bunu açıkça sordu). Ayrıca iki eşikli gri bölge
  (rule-out / belirsiz / rule-in) hesaplanır.
- Aux regresyon başlığı yalnızca eğitim sinyali olarak korunur; çıkarımda ölçüm yok.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds                                                    # noqa: E402
from dataset import EchoClipDataset, compute_aux_stats, AUX_COLS        # noqa: E402
from model import DiastolicModel, corn_loss, corn_predict, masked_mse   # noqa: E402
from train import set_seed, make_train_sampler                          # noqa: E402

LAP = ['normal', 'elevated']
SPLITS = os.path.expanduser('~/mimic-echo/splits_binary.csv')
MANIFEST = os.path.expanduser('~/mimic-echo/manifest_study.csv')


def study_probs(model, loader, dev, aux_w, max_batches=None):
    """Klip olasılıklarını çalışma düzeyinde ortala. Döner (study_id, y, p) listeleri + val loss."""
    model.eval()
    s_p, s_n, s_y = {}, {}, {}
    tot, nb = 0.0, 0
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x = b['x'].to(dev, non_blocking=True); y = b['grade'].to(dev)
            aux = b['aux'].to(dev); am = b['aux_mask'].to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ol, ap = model(x)
            tot += (corn_loss(ol.float(), y, 2) + aux_w * masked_mse(ap.float(), aux, am)).item(); nb += 1
            _, cum = corn_predict(ol.float())
            p = cum.float().cpu().numpy()[:, 0]                 # P(y>0) = P(yüksek LAP)
            for i, sid in enumerate(b['study_id']):
                s_p[sid] = s_p.get(sid, 0.0) + float(p[i]); s_n[sid] = s_n.get(sid, 0) + 1
                s_y[sid] = int(b['grade'][i])
    sids = sorted(s_p)
    return sids, np.array([s_y[s] for s in sids]), np.array([s_p[s] / s_n[s] for s in sids]), tot / max(nb, 1)


def binary_metrics(y, p, thr=None):
    m = {'n': int(len(y)), 'olay': int(y.sum()), 'prevalans': round(float(y.mean()), 4),
         'auroc': round(float(roc_auc_score(y, p)), 4),
         'auprc': round(float(average_precision_score(y, p)), 4),
         'brier': round(float(brier_score_loss(y, p)), 4)}
    if thr is not None:
        pr = (p >= thr).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
        tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        m.update(esik=round(float(thr), 4), TP=tp, FP=fp, TN=tn, FN=fn,
                 duyarlilik=round(tp / max(tp + fn, 1), 4), ozgulluk=round(tn / max(tn + fp, 1), 4),
                 PPD=round(tp / max(tp + fp, 1), 4), NPD=round(tn / max(tn + fn, 1), 4))
    return m


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(y, p):
    """Val üzerinde Platt ölçekleme. Sıralamayı değiştirmez, olasılıkları düzeltir."""
    lr = LogisticRegression(C=1e6, solver='lbfgs')
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p, ab):
    a, b = ab
    z = a * _logit(p) + b
    return 1.0 / (1.0 + np.exp(-z))


def calib_stats(y, p):
    """Kalibrasyon eğimi ve kesişimi. İdeal: eğim 1, kesişim 0.
    Eğim < 1 = aşırı uç tahminler; kesişim != 0 = sistematik kayma (calibration-in-the-large)."""
    z = _logit(p).reshape(-1, 1)
    sl = LogisticRegression(C=1e6, solver='lbfgs').fit(z, y)
    ic = LogisticRegression(C=1e6, solver='lbfgs', fit_intercept=True)
    ic.fit(np.zeros_like(z), y)                       # yalnız kesişim, offset olarak logit(p)
    slope = float(sl.coef_[0][0])
    inter = float(np.mean(y) - np.mean(p))            # basit ortalama fark (gözlenen - beklenen)
    bins = []
    q = np.quantile(p, np.linspace(0, 1, 6))
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() >= 5:
            bins.append({'beklenen': round(float(p[m].mean()), 4),
                         'gozlenen': round(float(y[m].mean()), 4), 'n': int(m.sum())})
    return {'egim': round(slope, 4), 'ortalama_fark': round(inter, 4), 'binler': bins}


def youden(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def gray_zone(y, p, sens=0.90, spec=0.90):
    """Rule-out esigi (>=%90 duyarlilik) ve rule-in esigi (>=%90 ozgulluk).

    DIKKAT: roc_curve esikleri AZALAN sirada verir ve ilk eleman +inf'tir. Ilk surumde
    inf ayiklanmadigi icin rule-in esigi sonsuz cikti ve test setinin %99,5'i gri bolgede
    gorundu. Duzeltilmis hali: sonlu esikler arasindan, duyarliligi saglayan en YUKSEK
    esik rule-out sinirini, ozgullugu saglayan en DUSUK esik rule-in sinirini verir."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    fpr, tpr, thr = fpr[ok], tpr[ok], thr[ok]
    lo = float(thr[tpr >= sens].max()) if (tpr >= sens).any() else float(np.min(p))
    hi = float(thr[(1 - fpr) >= spec].min()) if ((1 - fpr) >= spec).any() else float(np.max(p))
    return min(lo, hi), max(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.expanduser('~/mimic-echo/runs/b1_binary'))
    ap.add_argument('--splits', default=SPLITS)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--backbone-lr-mult', type=float, default=0.02,
                    help='ilk koşuda 0.1 ile en iyi epoch 0 çıktı (anında aşırı uyum); yumuşatıldı')
    ap.add_argument('--freeze-epochs', type=int, default=3,
                    help='ilk N epoch backbone dondurulur, yalnız başlıklar eğitilir')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--dropout', type=float, default=0.25)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--arch', default='panecho', choices=['panecho', 'panecho2d', 'r2plus1d'])
    ap.add_argument('--pretrained', type=int, default=1)
    ap.add_argument('--views', default='', help='virgüllü görünüm listesi; boş = filtre yok')
    ap.add_argument('--extra-dirs', default='', help='ek klip dizinleri (virgüllü)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = 'cuda'; set_seed(a.seed)

    ds.GRADES[:] = LAP                                   # etiket sözlüğü: 0=normal, 1=yüksek LAP
    aux_stats = compute_aux_stats(a.splits, MANIFEST)
    views = set(a.views.split(',')) if a.views else None
    extra = tuple(d for d in a.extra_dirs.split(',') if d)
    mk = lambda sp, tr_: EchoClipDataset(sp, a.clip_len, train=tr_, aux_stats=aux_stats,
                                         splits_csv=a.splits, manifest_csv=MANIFEST,
                                         extra_dirs=extra, views=views)
    tr, va, te = mk('train', True), mk('val', False), mk('test', False)
    print(f"[veri] train={len(tr)} val={len(va)} test={len(te)} klip", flush=True)

    tl = DataLoader(tr, batch_size=a.batch, sampler=make_train_sampler(tr, 2), num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    tel = DataLoader(te, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = DiastolicModel(a.clip_len, 2, len(AUX_COLS), pretrained=bool(a.pretrained),
                           dropout=a.dropout, arch=a.arch).to(dev)
    head = list(model.ord_head.parameters()) + list(model.aux_head.parameters())
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': a.lr * a.backbone_lr_mult},
                             {'params': head, 'lr': a.lr}], weight_decay=a.weight_decay)
    steps_ep = 8 if a.smoke else len(tl)
    total = steps_ep * (2 if a.smoke else a.epochs)
    warm = max(1, int(total * a.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total - warm), 1.0))))

    hist = []; best = -1.0; best_ep = -1; bad = 0; t0 = time.time()
    for ep in range(2 if a.smoke else a.epochs):
        frozen = ep < a.freeze_epochs
        for prm in model.backbone.parameters():
            prm.requires_grad = not frozen
        if ep == 0 or ep == a.freeze_epochs:
            print(f"  [backbone {'DONDURULDU' if frozen else 'çözüldü'}] ep{ep}", flush=True)
        model.train()
        for bi, b in enumerate(tl):
            if a.smoke and bi >= steps_ep:
                break
            x = b['x'].to(dev, non_blocking=True); y = b['grade'].to(dev)
            aux = b['aux'].to(dev); am = b['aux_mask'].to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ol, apr = model(x)
                loss = corn_loss(ol.float(), y, 2) + a.aux_weight * masked_mse(apr.float(), aux, am)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if bi % 100 == 0:
                print(f"  ep{ep} adım{bi}/{steps_ep} loss={loss.item():.3f}", flush=True)
        _, yv, pv, vloss = study_probs(model, vl, dev, a.aux_weight, max_batches=8 if a.smoke else None)
        M = binary_metrics(yv, pv)
        hist.append({'ep': ep, 'val_loss': round(vloss, 4), **M})
        print(f"[ep{ep}] val_loss={vloss:.3f} AUROC={M['auroc']} AUPRC={M['auprc']} "
              f"Brier={M['brier']} | {(time.time()-t0)/60:.1f}dk", flush=True)
        if M['auroc'] > best:
            best, best_ep, bad = M['auroc'], ep, 0
            torch.save(model.state_dict(), os.path.join(a.out_dir, 'best.pt'))
            print(f"  ✓ yeni en iyi val AUROC={best} (ep{ep})", flush=True)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[erken durdurma] en iyi ep{best_ep} AUROC={best}", flush=True)
                break

    model.load_state_dict(torch.load(os.path.join(a.out_dir, 'best.pt'), weights_only=True))
    sv, yv, pv, _ = study_probs(model, vl, dev, a.aux_weight)
    st, yt, pt_raw, _ = study_probs(model, tel, dev, a.aux_weight)

    # --- KALİBRASYON: yalnız val'de fit edilir, test'e ve dış kohorta değiştirilmeden uygulanır.
    # Platt ölçekleme sıralamayı korur, dolayısıyla AUROC değişmez; düzelttiği şey olasılıkların
    # kendisi (Brier, kalibrasyon eğimi). Eşik ve gri bölge kalibre edilmiş ölçekte belirlenir.
    ab = fit_platt(yv, pv)
    pv_c, pt_c = apply_platt(pv, ab), apply_platt(pt_raw, ab)
    thr = youden(yv, pv_c)                                # EŞİK YALNIZCA VAL'DEN
    lo, hi = gray_zone(yv, pv_c)
    pt = pt_c
    res = {'best_epoch': best_ep, 'val_auroc_best': best,
           'kalibrasyon': {'yontem': 'Platt (yalniz val)', 'a': round(ab[0], 4), 'b': round(ab[1], 4)},
           'esik_kaynagi': 'yalnizca val (Youden), kalibre edilmis olcekte', 'esik': round(thr, 4),
           'gri_bolge': [round(lo, 4), round(hi, 4)],
           'val_ham': binary_metrics(yv, pv), 'val': binary_metrics(yv, pv_c, thr),
           'test_ham': binary_metrics(yt, pt_raw), 'test': binary_metrics(yt, pt_c, thr),
           'kalibrasyon_val_once': calib_stats(yv, pv), 'kalibrasyon_val_sonra': calib_stats(yv, pv_c),
           'kalibrasyon_test_once': calib_stats(yt, pt_raw), 'kalibrasyon_test_sonra': calib_stats(yt, pt_c),
           'brier_taban_prevalans': round(float(yt.mean() * (1 - yt.mean())), 4),
           'hist': hist}
    gz = (pt >= lo) & (pt <= hi)
    res['test_gri_bolge'] = {'belirsiz_n': int(gz.sum()), 'belirsiz_oran': round(float(gz.mean()), 4)}
    if (~gz).sum() > 10:
        res['test_gri_disi'] = binary_metrics(yt[~gz], pt[~gz], thr)
    json.dump(res, open(os.path.join(a.out_dir, 'binary_results.json'), 'w'), indent=1, ensure_ascii=False)
    for name, sids, y, p, praw in (('val', sv, yv, pv_c, pv), ('test', st, yt, pt_c, pt_raw)):
        with open(os.path.join(a.out_dir, f'{name}_predictions.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['study_id', 'y_true', 'p_elevated', 'p_ham'])
            w.writerows([[s, int(yy), f'{pp:.5f}', f'{pr:.5f}'] for s, yy, pp, pr in zip(sids, y, p, praw)])
    print(json.dumps({k: res[k] for k in ('best_epoch', 'kalibrasyon', 'esik', 'gri_bolge',
                                          'val', 'test', 'kalibrasyon_test_once',
                                          'kalibrasyon_test_sonra', 'brier_taban_prevalans')},
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
