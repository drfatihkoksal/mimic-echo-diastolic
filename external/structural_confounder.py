#!/usr/bin/env python3
"""
Yapısal karıştırıcı analizi.

Hakem itirazı (JASE Hakem 2): B-mode görüntü, diyastolik fonksiyon dışında prognostik
olarak anlamlı çok şey içerir — hipertrofi, boşluk yeniden şekillenmesi, atriyum
büyüklüğü. EF'den bağımsız olmak, modelin diyastolik fonksiyona ÖZGÜ bir şey öğrendiğini
kanıtlamaz.

Sorulan üç soru:
  (1) Yapısal ölçümler TEK BAŞINA hedefi ne kadar öngörüyor?
  (2) Model skoru yapısal modele bir şey EKLİYOR mu? (eşleştirilmiş bootstrap ΔAUROC)
  (3) Model skorunun ne kadarı yapısal ölçümlerle AÇIKLANIYOR? (logit(skor) ~ yapı, R²)

Dikkat: LAVi, 2025 ASE algoritmasının ikincil değişkenidir; bazı çalışmalarda etiketin
kendisine girer. Bu yüzden analiz LAVi'li ve LAVi'siz olmak üzere iki kez yapılır ve
LAVi'siz sürüm asıl cevap sayılır.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260901
N_BOOT = 4000
H = os.path.expanduser('~/mimic-echo/')


def logit(p, e=1e-6):
    p = np.clip(np.asarray(p, float), e, 1 - e)
    return np.log(p / (1 - p))


def load(cols):
    lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
    y, s, X = [], [], []
    for r in csv.DictReader(open(H + 'runs/b2_binary/test_predictions.csv')):
        l = lab.get(r['study_id'])
        if not l:
            continue
        row = []
        ok = True
        for c in cols:
            v = l.get(c)
            if c == 'gender':
                row.append(1.0 if str(v).upper().startswith('F') else 0.0); continue
            if v in ('', 'None', None):
                ok = False; break
            row.append(float(v))
        if not ok:
            continue
        y.append(int(r['y_true'])); s.append(float(r['p_elevated'])); X.append(row)
    return np.array(y), np.array(s), np.array(X)


def fit_auc(X, y):
    m = LogisticRegression(max_iter=2000).fit(X, y)
    return m, m.predict_proba(X)[:, 1]


def paired_delta(y, p_struct, p_both, rng, n=N_BOOT):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p_both[i]) - roc_auc_score(y[i], p_struct[i]))
    d = np.array(d)
    return (round(float(d.mean()), 4),
            [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            round(float(np.mean(d <= 0)), 4))


def boot_auc(y, p, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run(name, cols):
    y, s, X = load(cols)
    if len(y) < 50:
        return {'ad': name, 'n': int(len(y)), 'not': 'yetersiz'}
    Xz = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1, X.std(0))
    _, p_struct = fit_auc(Xz, y)
    both = np.column_stack([Xz, logit(s)])
    m_both, p_both = fit_auc(both, y)
    rng = np.random.default_rng(SEED)
    au_s = roc_auc_score(y, p_struct); au_i = roc_auc_score(y, s); au_b = roc_auc_score(y, p_both)
    d, ci, pval = paired_delta(y, p_struct, p_both, np.random.default_rng(SEED))
    # skorun yapıyla açıklanan payı
    r2 = LinearRegression().fit(Xz, logit(s)).score(Xz, logit(s))
    return {'ad': name, 'degiskenler': cols, 'n': int(len(y)), 'olay': int(y.sum()),
            'auroc_yapisal': round(au_s, 4), 'auroc_yapisal_ga': boot_auc(y, p_struct, np.random.default_rng(SEED)),
            'auroc_model': round(au_i, 4), 'auroc_model_ga': boot_auc(y, s, np.random.default_rng(SEED)),
            'auroc_birlesik': round(au_b, 4),
            'delta_model_katkisi': d, 'delta_ga': ci, 'p_delta_sifir_veya_negatif': pval,
            'skorun_yapiyla_aciklanan_payi_R2': round(float(r2), 4),
            'duzeltilmis_OR_skor_logit_birimi': round(float(np.exp(m_both.coef_[0][-1])), 3)}


def main():
    res = {'aciklama': 'iç test seti; model skoru kalibre olasılık', 'tohum': SEED, 'analizler': []}
    for name, cols in [
        ('yapı: LVMi + yaş + cinsiyet (LAVi HARİÇ — asıl analiz)', ['LVMi', 'age', 'gender']),
        ('yapı: LVMi + LAVi + yaş + cinsiyet (LAVi etiketin bileşeni)', ['LVMi', 'LAVi', 'age', 'gender']),
        ('yapı: LVMi + EF + yaş + cinsiyet (EF alt kümesi)', ['LVMi', 'EF', 'age', 'gender']),
    ]:
        r = run(name, cols); res['analizler'].append(r)
        if 'not' in r:
            print(name, r); continue
        print(f"\n== {name}  (n={r['n']}, olay={r['olay']})")
        print(f"   yapısal model AUROC = {r['auroc_yapisal']} {r['auroc_yapisal_ga']}")
        print(f"   görüntü modeli AUROC = {r['auroc_model']} {r['auroc_model_ga']}")
        print(f"   birleşik AUROC = {r['auroc_birlesik']}")
        print(f"   modelin katkısı ΔAUROC = {r['delta_model_katkisi']:+.4f} {r['delta_ga']}  p={r['p_delta_sifir_veya_negatif']}")
        print(f"   skorun yapıyla açıklanan payı R² = {r['skorun_yapiyla_aciklanan_payi_R2']}")
        print(f"   düzeltilmiş OR (skor logit birimi başına) = {r['duzeltilmis_OR_skor_logit_birimi']}")
    json.dump(res, open(os.path.join(HERE, 'structural_confounder.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n-> external/structural_confounder.json')


if __name__ == '__main__':
    main()
