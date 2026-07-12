#!/usr/bin/env python3
"""
Faz C — Ablation 1: SADECE-EF baseline (klinik referans).

Kilit soru: görüntü modeli EF'in (sistolik fonksiyon) ötesinde gerçek DİYASTOLİK sinyal mi öğreniyor?
EF tek başına diyastolik grade'i ne kadar tahmin eder? Model bunu belirgin aşıyorsa → diyastolik-özel
bilgi öğreniyor demektir.

Yöntem: train study'lerinde EF→grade (multinomial lojistik, class_weight=balanced, özellik [EF,EF²]).
Test'te EF'i OLAN study'lerde değerlendir. ADİL karşılaştırma: C.1 modelinin (test_predictions.csv)
tahminlerini AYNI alt-kümede yeniden skorla → head-to-head.

Kullanım: python3 ablation_ef.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, confusion_matrix, f1_score

GRADES = ['Normal', 'Grade1', 'Grade2', 'Grade3']
SPLITS = os.path.expanduser('~/mimic-echo/splits.csv')
MANIFEST = os.path.expanduser('~/mimic-echo/manifest_study.csv')
C1_PRED = os.path.expanduser('~/mimic-echo/runs/c1_panecho/test_predictions.csv')


def metrics(t, p, k):
    """t,p: gerçek/tahmin ordinal diziler; k: sınıf sayısı (4 veya 3)."""
    lab = list(range(k))
    return {
        'n': int(len(t)),
        'qwk': round(float(cohen_kappa_score(t, p, weights='quadratic', labels=lab)), 4),
        'acc': round(float((t == p).mean()), 4),
        'acc_pm1': round(float((np.abs(t - p) <= 1).mean()), 4),
        'bacc': round(float(balanced_accuracy_score(t, p)), 4),
        'macro_f1': round(float(f1_score(t, p, average='macro', labels=lab, zero_division=0)), 4),
        'confusion': confusion_matrix(t, p, labels=lab).tolist(),
    }


def load_rows():
    ef, grade = {}, {}
    with open(MANIFEST, newline='') as f:
        for r in csv.DictReader(f):
            try: ef[r['study_id']] = float(r['EF'])
            except Exception: pass
            if r['grade_2025'] in GRADES: grade[r['study_id']] = GRADES.index(r['grade_2025'])
    split = {}
    with open(SPLITS, newline='') as f:
        for r in csv.DictReader(f): split[r['study_id']] = r['split']
    return ef, grade, split


def main():
    ef, grade, split = load_rows()

    def xy(which):
        sids = [s for s in split if split[s] == which and s in ef and s in grade]
        X = np.array([[ef[s], ef[s] ** 2] for s in sids], dtype=np.float64)
        y = np.array([grade[s] for s in sids])
        return sids, X, y

    _, Xtr, ytr = xy('train')
    te_sids, Xte, yte = xy('test')
    print(f"EF-baseline | train(EF+grade)={len(ytr)} | test(EF+grade)={len(yte)} "
          f"(438 test'in {100*len(yte)/438:.0f}%'i EF'li)")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))

    out = {'ef_baseline': {}, 'c1_same_subset': {}}
    print("\n=== SADECE-EF baseline (test, EF'li alt-küme) ===")
    for name, k, tt, pp in [('4-sınıf', 4, yte, pred),
                            ('3-sınıf', 3, np.minimum(yte, 2), np.minimum(pred, 2))]:
        m = metrics(tt, pp, k); out['ef_baseline'][name] = m
        print(f"[{name}] QWK={m['qwk']} acc={m['acc']} ±1={m['acc_pm1']} bacc={m['bacc']} macroF1={m['macro_f1']}")

    # --- C.1 modelini AYNI alt-kümede yeniden skorla ---
    c1 = {}
    with open(C1_PRED, newline='') as f:
        for r in csv.DictReader(f):
            c1[r['study_id']] = (GRADES.index(r['true']), GRADES.index(r['pred']))
    sub = [s for s in te_sids if s in c1]
    ct = np.array([c1[s][0] for s in sub]); cp = np.array([c1[s][1] for s in sub])
    print(f"\n=== C.1 modeli (AYNI EF'li alt-küme, n={len(sub)}) — head-to-head ===")
    for name, k, tt, pp in [('4-sınıf', 4, ct, cp),
                            ('3-sınıf', 3, np.minimum(ct, 2), np.minimum(cp, 2))]:
        m = metrics(tt, pp, k); out['c1_same_subset'][name] = m
        print(f"[{name}] QWK={m['qwk']} acc={m['acc']} ±1={m['acc_pm1']} bacc={m['bacc']} macroF1={m['macro_f1']}")

    outdir = os.path.expanduser('~/mimic-echo/runs/ablation_ef'); os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, 'metrics.json'), 'w'), indent=1)
    ef3, c13 = out['ef_baseline']['3-sınıf']['qwk'], out['c1_same_subset']['3-sınıf']['qwk']
    print(f"\n>>> 3-sınıf QWK: EF-only={ef3}  vs  C.1(görüntü)={c13}  "
          f"(fark {c13-ef3:+.3f} → görüntü modeli EF'in ötesinde {'diyastolik sinyal öğreniyor' if c13>ef3+0.03 else 'BELİRGİN katkı yok'})")


if __name__ == '__main__':
    main()
