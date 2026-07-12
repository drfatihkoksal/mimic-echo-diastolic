#!/usr/bin/env python3
"""
Faz C — Prognostik analiz: 1-yıl all-cause mortalite (MIMIC-IV dod).

Bağlantı: echo-record-list.acquisition_datetime (echo zamanı) ↔ patients.dod (ölüm),
ikisi de aynı hasta-bazlı kaydırılmış zaman çizgisinde → echo'dan ölüme gün. Endpoint:
1-yıl all-cause mortalite (olay = ≤365g ölüm; aksi 365g'de sansür → temiz 1-yıl KM).

A) REFERANS grade prognostik mi? — KM + log-rank + Cox (grade+yaş+cinsiyet+EF).
B) MODEL tahmini prognostik mi? — model risk skoru P(≥G2) → KM + Cox + C-index;
   EF'den BAĞIMSIZ mı (yaş/cinsiyet/EF'e göre düzeltilmiş)?

Tahmin kaynağı (--preds):
  * test_predictions.csv (varsayılan): sadece TEST seti (n=438, ~37 ölüm — az güçlü).
  * oof_predictions.csv: 5-kat OOF → TÜM kohort yansız tahminli (n=3065, ~300 ölüm).

Bağımsızlık: bir hastanın birden fazla echo'su varsa aynı ölüm defalarca sayılır. Bu yüzden
BİRİNCİL analiz hasta başına TEK echo (en erken; --unit first), DUYARLILIK analizi tüm
study'ler + hasta-kümesi-robust SE (--unit all). KM/log-rank yalnızca `first` altında
geçerlidir; `all` altında bunlar tanımlayıcıdır, çıkarım Cox-robust'a dayanır.

Kullanım:
  python3 mortality_analysis.py                                    # test preds, first-echo
  python3 mortality_analysis.py --preds ~/mimic-echo/runs/oof_predictions.csv
  python3 mortality_analysis.py --preds .../oof_predictions.csv --unit all
"""
from __future__ import annotations
import argparse, csv, os
from datetime import datetime
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index

ROOT = os.environ.get("MIMIC_IV_DIR", "")   # export MIMIC_IV_DIR=/yol/mimic-iv-3.1
GR = ['Normal', 'Grade1', 'Grade2', 'Grade3']
HORIZON = 365
DEFAULT_PREDS = os.path.expanduser('~/mimic-echo/runs/c1_panecho/test_predictions.csv')


def d(s): return datetime.strptime(s[:10], '%Y-%m-%d')


def build_df():
    split = {}
    with open(os.path.expanduser('~/mimic-echo/splits.csv')) as f:
        for r in csv.DictReader(f): split[r['study_id']] = r['split']
    coh = {}
    with open(os.path.expanduser('~/mimic-echo/manifest_study.csv')) as f:
        for r in csv.DictReader(f):
            if r['study_id'] in split and r['grade_2025'] in GR:
                coh[r['study_id']] = dict(subject=r['subject_id'], grade=GR.index(r['grade_2025']),
                                          split=split[r['study_id']], ef=r['EF'], age=r['age'], gender=r['gender'])
    subs = set(v['subject'] for v in coh.values())
    acq = {}
    with open(os.path.expanduser('~/mimic-echo/echo-record-list.csv')) as f:
        for r in csv.DictReader(f):
            if r['study_id'] in coh:
                t = r['acquisition_datetime']
                if r['study_id'] not in acq or t < acq[r['study_id']]: acq[r['study_id']] = t
    dod = {}
    with open(os.path.join(ROOT, 'hosp', 'patients.csv')) as f:
        for r in csv.DictReader(f):
            if r['subject_id'] in subs: dod[r['subject_id']] = r['dod']
    rows = []
    for sid, v in coh.items():
        if sid not in acq: continue
        a = d(acq[sid]); dd = dod.get(v['subject'], '')
        if dd:
            days = (d(dd) - a).days
            event = 1 if 0 <= days <= HORIZON else 0
            time = min(days, HORIZON) if days >= 0 else 0
        else:
            event = 0; time = HORIZON
        try: ef = float(v['ef'])
        except Exception: ef = np.nan
        try: age = float(v['age'])
        except Exception: age = np.nan
        rows.append(dict(study_id=sid, subject=v['subject'], grade=v['grade'], split=v['split'],
                         ef=ef, age=age, female=1 if v['gender'].upper().startswith('F') else 0,
                         acq=acq[sid], time=time, event=event))
    return pd.DataFrame(rows)


def load_preds(path):
    """test_predictions.csv (cum_probs='a;b;c') veya oof_predictions.csv (p_ge1/2/3) — ikisini de oku."""
    pred = {}
    with open(os.path.expanduser(path), newline='') as f:
        for r in csv.DictReader(f):
            if 'cum_probs' in r:
                cum = [float(x) for x in r['cum_probs'].split(';')]
            else:
                cum = [float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])]
            pred[r['study_id']] = dict(pred_grade=GR.index(r['pred']), p_ge1=cum[0], p_ge2=cum[1])
    return pred


def first_echo(df):
    """Hasta başına en erken echo → bağımsız gözlemler (KM/log-rank için gerekli)."""
    return df.sort_values('acq').groupby('subject', as_index=False).first()


def cox_fit(data, cluster=None):
    """cluster verilirse hasta-kümesi-robust SE (tekrarlı study'ler için)."""
    cph = CoxPHFitter()
    if cluster:
        cph.fit(data, 'time', 'event', cluster_col=cluster, robust=True)
    else:
        cph.fit(data, 'time', 'event')
    return cph


def hr_str(cph, var):
    hr = cph.hazard_ratios_[var]; ci = cph.confidence_intervals_.loc[var]
    p = cph.summary.loc[var, 'p']
    return f"{hr:.2f} [{np.exp(ci.iloc[0]):.2f}-{np.exp(ci.iloc[1]):.2f}], p={p:.2e}"


def analyze_reference(df, clust):
    print("═══ A) REFERANS grade — 1-yıl mortalite ═══")
    print("1-yıl ölüm oranı:")
    for g in range(4):
        s = df[df.grade == g]
        print(f"  {GR[g]:8s}: {s.event.sum():3d}/{len(s):4d} = {100*s.event.mean():.1f}%")
    lr = multivariate_logrank_test(df.time, df.grade, df.event)
    print(f"log-rank (4 grup) p = {lr.p_value:.2e}")
    df = df.assign(grade3=df.grade.clip(upper=2))
    lr3 = multivariate_logrank_test(df.time, df.grade3, df.event)
    print(f"log-rank (3 grup N/G1/G2-3) p = {lr3.p_value:.2e}")
    cols = ['time', 'event', 'grade', 'age', 'female'] + ([clust] if clust else [])
    cox = cox_fit(df[cols].dropna(), clust)
    print(f"Cox HR (grade başına artış, yaş+cinsiyet düzeltmeli): {hr_str(cox, 'grade')}")
    cols_ef = ['time', 'event', 'grade', 'age', 'female', 'ef'] + ([clust] if clust else [])
    dfe = df[cols_ef].dropna()
    coxe = cox_fit(dfe, clust)
    print(f"Cox HR (grade, +EF düzeltmeli, n={len(dfe)}): {hr_str(coxe, 'grade')}  → EF'e rağmen prognostik")


def analyze_model(df, clust, label):
    print(f"\n═══ B) MODEL tahmini — 1-yıl mortalite ({label}) ═══")
    print(f"tahminli study: {len(df)} | 1-yıl ölüm: {df.event.sum()} ({100*df.event.mean():.1f}%)")
    df = df.assign(pred_adv=(df.pred_grade >= 2).astype(int))
    for lbl, s in [('tahmin: Normal/G1', df[df.pred_adv == 0]), ('tahmin: ileri-DD (≥G2)', df[df.pred_adv == 1])]:
        print(f"  {lbl:22s}: {s.event.sum():3d}/{len(s):4d} = {100*s.event.mean():.1f}%")
    a, b = df[df.pred_adv == 1], df[df.pred_adv == 0]
    lrp = logrank_test(a.time, b.time, a.event, b.event)
    print(f"  log-rank (tahmini ileri-DD) p = {lrp.p_value:.2e}")
    # tahmini grade'ler arası (4 grup)
    lr4 = multivariate_logrank_test(df.time, df.pred_grade, df.event)
    print(f"  log-rank (tahmini 4 grade) p = {lr4.p_value:.2e}")
    # ayrıştırma gücü: C-index
    cidx_model = concordance_index(df.time, -df.risk, df.event)
    de = df.dropna(subset=['ef'])
    cidx_ef = concordance_index(de.time, de.ef, de.event)
    print(f"  C-index: model risk={cidx_model:.3f} | EF tek başına={cidx_ef:.3f} (n={len(de)})")
    # EF'den bağımsız mı: Cox risk + yaş + cinsiyet + EF
    cols = ['time', 'event', 'risk', 'age', 'female', 'ef'] + ([clust] if clust else [])
    dm = df[cols].dropna()
    if dm.event.sum() >= 10:
        coxm = cox_fit(dm, clust)
        print(f"  Cox HR model-risk (yaş+cinsiyet+EF düzeltmeli, n={len(dm)}, "
              f"{dm.event.sum()} olay): {hr_str(coxm, 'risk')}")
        # EF+yaş+cinsiyet modeline model-riskinin KATKISI (C-index karşılaştırma)
        base = cox_fit(dm.drop(columns=['risk']), clust)
        c_base = concordance_index(dm.time, -base.predict_partial_hazard(dm), dm.event)
        c_full = concordance_index(dm.time, -coxm.predict_partial_hazard(dm), dm.event)
        print(f"  C-index: EF+yaş+cinsiyet={c_base:.3f} → +model-risk={c_full:.3f} (Δ={c_full-c_base:+.3f})")
    else:
        print(f"  (yalnız {dm.event.sum()} olay — Cox atlandı, güç yetersiz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=DEFAULT_PREDS,
                    help='test_predictions.csv (yalnız test) veya oof_predictions.csv (tüm kohort)')
    ap.add_argument('--unit', choices=['first', 'all'], default='first',
                    help='first: hasta başına en erken echo (bağımsız gözlem, birincil). '
                         'all: tüm study + hasta-kümesi-robust SE (duyarlılık)')
    ap.add_argument('--out', default='~/mimic-echo/runs/mortality_cohort.csv')
    args = ap.parse_args()

    full = build_df()
    pred = load_preds(args.preds)
    full['pred_grade'] = full.study_id.map(lambda s: pred.get(s, {}).get('pred_grade', np.nan))
    full['risk'] = full.study_id.map(lambda s: pred.get(s, {}).get('p_ge2', np.nan))
    is_oof = 'oof' in os.path.basename(args.preds).lower()

    # Model analizinin evreni: OOF → tüm kohort (yansız); aksi → yalnız test seti
    scope = full if is_oof else full[full.split == 'test']
    scope = scope.dropna(subset=['pred_grade', 'risk'])

    if args.unit == 'first':
        ref_df, mod_df, clust = first_echo(full), first_echo(scope), None
        unit_note = "hasta başına en erken echo (bağımsız gözlemler)"
    else:
        ref_df, mod_df, clust = full, scope, 'subject'
        unit_note = "tüm study'ler + hasta-kümesi-robust SE (tekrarlı ölçüm)"

    print(f"Tahmin kaynağı: {args.preds}")
    print(f"Analiz birimi : {unit_note}")
    print(f"Kohort (referans): {len(ref_df)} gözlem / {ref_df.subject.nunique()} hasta | "
          f"1-yıl ölüm: {ref_df.event.sum()} ({100*ref_df.event.mean():.1f}%)")
    print(f"Kohort (model)   : {len(mod_df)} gözlem / {mod_df.subject.nunique()} hasta | "
          f"1-yıl ölüm: {mod_df.event.sum()} ({100*mod_df.event.mean():.1f}%)\n")

    analyze_reference(ref_df, clust)
    analyze_model(mod_df, clust, 'OOF — tüm kohort, yansız' if is_oof else 'yalnız TEST seti')

    out = os.path.expanduser(args.out)
    full.to_csv(out, index=False)
    print(f"\nKaydedildi: {out} (KM figürleri için; pred_grade/risk kolonlarıyla)")


if __name__ == '__main__':
    main()
