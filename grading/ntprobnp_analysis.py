#!/usr/bin/env python3
"""
Faz C — ECHO-DIŞI doğrulama: NT-proBNP (MIMIC-IV labevents, itemid 50963).

Neden kritik: referans etiketlerimiz AYNI echo'nun structured measurement'larından türetildi.
Bu yüzden "model sadece etiket sızıntısını öğrendi" itirazı yapılabilir. NT-proBNP KANDAN
ölçülür — echo'dan tamamen bağımsız bir modalite ve yükselmiş dolum basıncının/duvar
geriliminin yerleşik biyobelirteci. Model riski NT-proBNP ile ilişkiliyse, B-mode'dan
öğrenilen sinyal echo-dışı bir kaynakla doğrulanmış olur.

A) REFERANS grade ↔ NT-proBNP: biyolojik gradyan (etiket geçerliliği, echo-dışı kanıt).
B) MODEL tahmini ↔ NT-proBNP: sinyal gerçek mi? EF'den BAĞIMSIZ mı?
   (log-NTproBNP ~ model_risk + yaş + cinsiyet + EF çoklu regresyonu)
   + yükselmiş NT-proBNP'yi saptamada model riskinin AUROC'u.

Eşleştirme: echo tarihine EN YAKIN ölçüm, |Δ| ≤ --window gün (varsayılan 7).
Bağımsızlık: hasta başına tek study (en erken echo) — bkz. mortality_analysis.py.

ÖNEMLİ KISIT: NT-proBNP ölçülen hastalar seçilmiş bir alt kümedir (daha ağır, HF şüphesi
olan hastalar). Sonuçlar bu alt küme için geçerlidir, tüm kohorta genellenemez.

Kullanım:
  python3 ntprobnp_analysis.py --preds ~/mimic-echo/runs/oof_predictions.csv
"""
from __future__ import annotations
import argparse, csv, os, collections
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

GR = ['Normal', 'Grade1', 'Grade2', 'Grade3']
H = os.path.expanduser('~/mimic-echo/')
RAW = os.path.join(H, 'ntprobnp_raw.csv')     # subject_id,charttime,valuenum (itemid 50963)


def _dt(s): return datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')
def _d(s):  return datetime.strptime(s[:10], '%Y-%m-%d')


def build(window, preds_path):
    sp = {r['study_id']: r for r in csv.DictReader(open(os.path.join(H, 'splits.csv')))}
    man = {r['study_id']: r for r in csv.DictReader(open(os.path.join(H, 'manifest_study.csv')))}
    acq = {}
    for r in csv.DictReader(open(os.path.join(H, 'echo-record-list.csv'))):
        s = r['study_id']
        if s in sp and (s not in acq or r['acquisition_datetime'] < acq[s]):
            acq[s] = r['acquisition_datetime']

    subs = {r['subject_id'] for r in sp.values()}
    labs = collections.defaultdict(list)
    for line in open(RAW):
        s, t, v = line.rstrip('\n').split(',')
        if s in subs and v:
            try: labs[s].append((_dt(t), float(v)))
            except ValueError: pass

    pred = {}
    with open(os.path.expanduser(preds_path), newline='') as f:
        for r in csv.DictReader(f):
            cum = ([float(x) for x in r['cum_probs'].split(';')] if 'cum_probs' in r
                   else [float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])])
            pred[r['study_id']] = (GR.index(r['pred']), cum[1])   # (tahmin grade, P(≥G2)=risk)

    rows = []
    for sid, r in sp.items():
        if sid not in acq:
            continue
        a = _d(acq[sid]); best = None
        for t, v in labs.get(r['subject_id'], []):
            gap = abs((t - a).days)
            if gap <= window and (best is None or gap < best[0]):
                best = (gap, v)
        if best is None or best[1] <= 0:
            continue
        m = man.get(sid, {})
        f = lambda k: (float(m[k]) if m.get(k) not in (None, '', 'NA') else np.nan)
        p = pred.get(sid)
        rows.append(dict(study_id=sid, subject=r['subject_id'], acq=acq[sid],
                         grade=GR.index(r['grade_2025']), bnp=best[1], gap=best[0],
                         age=f('age'), female=1 if m.get('gender', '').upper().startswith('F') else 0,
                         ef=f('EF'),
                         pred_grade=(p[0] if p else np.nan), risk=(p[1] if p else np.nan)))
    return rows


def first_echo(rows):
    """hasta başına en erken echo → bağımsız gözlemler."""
    by = {}
    for r in sorted(rows, key=lambda x: x['acq']):
        by.setdefault(r['subject'], r)
    return list(by.values())


def med_iqr(v):
    return f"{np.median(v):7.0f} [{np.percentile(v,25):5.0f}-{np.percentile(v,75):6.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='~/mimic-echo/runs/oof_predictions.csv')
    ap.add_argument('--window', type=int, default=7, help='echo↔lab en fazla kaç gün ara')
    ap.add_argument('--unit', choices=['first', 'all'], default='first')
    ap.add_argument('--out', default='~/mimic-echo/runs/ntprobnp_cohort.csv')
    args = ap.parse_args()

    rows = build(args.window, args.preds)
    if args.unit == 'first':
        rows = first_echo(rows)
    print(f"Tahmin kaynağı: {args.preds}")
    print(f"Eşleşme: |echo − lab| ≤ {args.window} gün | birim: {args.unit}")
    print(f"n = {len(rows)} study / {len({r['subject'] for r in rows})} hasta "
          f"| medyan gecikme {np.median([r['gap'] for r in rows]):.0f} gün\n")

    bnp = np.array([r['bnp'] for r in rows]); lb = np.log(bnp)
    g = np.array([r['grade'] for r in rows])

    # ═══ A) REFERANS grade ↔ NT-proBNP (etiket geçerliliği, echo-DIŞI) ═══
    print("═══ A) REFERANS grade ↔ NT-proBNP (pg/mL, medyan [IQR]) ═══")
    groups = []
    for k in range(4):
        v = bnp[g == k]
        if len(v):
            groups.append(v)
            print(f"  {GR[k]:8s} n={len(v):4d}: {med_iqr(v)}")
    if len(groups) > 1:
        H_, p = stats.kruskal(*groups)
        rho, prho = stats.spearmanr(g, lb)
        print(f"  Kruskal-Wallis p = {p:.2e} | Spearman(grade, log-BNP) ρ = {rho:.3f} (p={prho:.2e})")

    # ═══ B) MODEL tahmini ↔ NT-proBNP (echo-DIŞI doğrulama) ═══
    ok = [r for r in rows if not np.isnan(r['risk'])]
    if not ok:
        print("\n(model tahmini yok — --preds dosyası bu study'leri kapsamıyor)")
        return
    print(f"\n═══ B) MODEL tahmini ↔ NT-proBNP (n={len(ok)}) ═══")
    pg = np.array([r['pred_grade'] for r in ok]); rk = np.array([r['risk'] for r in ok])
    b2 = np.array([r['bnp'] for r in ok]); lb2 = np.log(b2)
    for k in range(4):
        v = b2[pg == k]
        if len(v):
            print(f"  tahmin {GR[k]:8s} n={len(v):4d}: {med_iqr(v)}")
    rho, prho = stats.spearmanr(rk, lb2)
    print(f"  Spearman(model riski P(≥G2), log-BNP) ρ = {rho:.3f} (p={prho:.2e})")

    # yükselmiş NT-proBNP'yi saptama: model riskinin AUROC'u (yerleşik eşikler)
    print("  Model riskinin yükselmiş NT-proBNP'yi saptama AUROC'u:")
    for thr in (125, 300, 1000):
        y = (b2 > thr).astype(int)
        if 0 < y.sum() < len(y):
            print(f"    > {thr:4d} pg/mL (pozitif {y.sum():4d}/{len(y)}): AUROC = {roc_auc_score(y, rk):.3f}")

    # EF'den BAĞIMSIZ mı: log-BNP ~ risk + yaş + cinsiyet + EF
    sub = [r for r in ok if not any(np.isnan([r['age'], r['ef']]))]
    if len(sub) >= 30:
        X = np.column_stack([[r['risk'] for r in sub], [r['age'] for r in sub],
                             [r['female'] for r in sub], [r['ef'] for r in sub],
                             np.ones(len(sub))])
        y = np.log([r['bnp'] for r in sub])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(sub) - X.shape[1]
        s2 = resid @ resid / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        t = beta / se
        pvals = 2 * stats.t.sf(np.abs(t), dof)
        names = ['model_risk', 'yaş', 'kadın', 'EF', 'sabit']
        print(f"\n  Çoklu regresyon: log(NT-proBNP) ~ model_risk + yaş + cinsiyet + EF (n={len(sub)})")
        for i, nm in enumerate(names[:-1]):
            star = ' ***' if pvals[i] < 0.001 else (' **' if pvals[i] < 0.01 else (' *' if pvals[i] < 0.05 else ''))
            print(f"    {nm:11s} β = {beta[i]:+.3f} (SE {se[i]:.3f}), p = {pvals[i]:.2e}{star}")
        # model_risk'in katkısı: EF'li modele göre R² artışı
        X0 = np.column_stack([[r['age'] for r in sub], [r['female'] for r in sub],
                              [r['ef'] for r in sub], np.ones(len(sub))])
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        r0 = y - X0 @ b0
        R2_0 = 1 - (r0 @ r0) / ((y - y.mean()) @ (y - y.mean()))
        R2_1 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
        print(f"    R²: yaş+cinsiyet+EF = {R2_0:.3f} → +model_risk = {R2_1:.3f} (Δ = {R2_1-R2_0:+.3f})")
        if pvals[0] < 0.05:
            print("    → model riski EF'den BAĞIMSIZ olarak NT-proBNP ile ilişkili (echo-dışı doğrulama)")

    out = os.path.expanduser(args.out)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nKaydedildi: {out}")


if __name__ == '__main__':
    main()
