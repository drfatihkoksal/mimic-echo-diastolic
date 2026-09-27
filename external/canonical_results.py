#!/usr/bin/env python3
"""
TEK KAYNAK: makalede raporlanan bütün sayılar buradan üretilir.

Gerekçe: güven aralıkları bootstrap ile hesaplanıyor ve farklı betikler farklı tohum
kullandığında aynı büyüklük için birbirinden hafifçe farklı aralıklar çıkıyordu
(ör. dış ASE LAP için 0,858-0,963 ile 0,863-0,962). Yayında bu kabul edilemez.
Bu modül tohumu ve tekrar sayısını sabitler, bütün kohortları ve metrikleri tek
geçişte üretir; şekiller ve metin bu JSON'u okur, yeniden hesaplamaz.

Kohort tanımları, eşik, kalibratör ve gri bölge external/PRESPECIFICATION.md'de
kilitlenmiştir; burada yeniden tanımlanmaz, yalnız uygulanır.
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'grading'))

SEED = 20260901          # sabit; değiştirilirse tüm rapor edilen aralıklar değişir
N_BOOT = 4000
OUT = os.path.join(HERE, 'canonical_results.json')


def logit(p, e=1e-6):
    p = np.clip(np.asarray(p, float), e, 1 - e)
    return np.log(p / (1 - p))


def cal_slope(y, p):
    return float(LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y).coef_[0][0])


def citl(y, p):
    z = logit(p); b = 0.0
    for _ in range(200):
        q = 1 / (1 + np.exp(-(z + b)))
        g = np.sum(y - q); h = np.sum(q * (1 - q))
        if h < 1e-9:
            break
        s = g / h; b += s
        if abs(s) < 1e-10:
            break
    return float(b)


def murphy(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, bins - 1)
    ybar = y.mean(); n = len(y); rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - ybar) ** 2
    return round(rel / n, 5), round(res / n, 5), round(float(ybar * (1 - ybar)), 5)


def boot_ci(y, p, fn, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        try:
            v.append(fn(y[i], p[i]))
        except Exception:
            pass
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def at_threshold(y, p, thr):
    pr = (p >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'duyarlilik': round(tp / max(tp + fn, 1), 4), 'ozgulluk': round(tn / max(tn + fp, 1), 4),
            'PPD': round(tp / max(tp + fp, 1), 4), 'NPD': round(tn / max(tn + fn, 1), 4)}


def describe(y, p, praw, thr, lo, hi, rng):
    rel, res, unc = murphy(y, p)
    d = {'n': int(len(y)), 'olay': int(y.sum()), 'prevalans': round(float(y.mean()), 4),
         'auroc': round(float(roc_auc_score(y, p)), 4),
         'auroc_ga': boot_ci(y, p, roc_auc_score, rng),
         'auprc': round(float(average_precision_score(y, p)), 4),
         'brier_ham': round(float(brier_score_loss(y, praw)), 4),
         'brier': round(float(brier_score_loss(y, p)), 4),
         'guvenilirlik': rel, 'cozunurluk': res, 'belirsizlik_taban': unc,
         'kalibrasyon_egimi': round(cal_slope(y, p), 3),
         'kalibrasyon_egimi_ga': boot_ci(y, p, cal_slope, rng),
         'kesisim': round(citl(y, p), 3), 'kesisim_ga': boot_ci(y, p, citl, rng),
         'ortalama_fark': round(float(y.mean() - p.mean()), 4),
         'esikte': at_threshold(y, p, thr)}
    d['beceri_puani'] = round(1 - d['brier'] / unc, 4)
    gz = (p > lo) & (p < hi)
    d['gri_bolge'] = {'belirsiz_n': int(gz.sum()), 'belirsiz_oran': round(float(gz.mean()), 4)}
    if (~gz).sum() > 20 and y[~gz].sum() >= 5:
        d['gri_disi'] = {'n': int((~gz).sum()), 'olay': int(y[~gz].sum()),
                         'auroc': round(float(roc_auc_score(y[~gz], p[~gz])), 4),
                         'auroc_ga': boot_ci(y[~gz], p[~gz], roc_auc_score, rng),
                         **at_threshold(y[~gz], p[~gz], thr)}
    return d


def cohorts():
    """Kohortların TEK tanımı. Başka betikler bunu import eder, yeniden kurmaz."""
    it = list(csv.DictReader(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/test_predictions.csv'))))
    ex = list(csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv'))))

    def pick(col):
        # AF dışlanmamış ve ilgili son noktası tanımlı muayeneler
        r = [x for x in ex if x[col] != '' and x['af_excluded'] == '0']
        return (np.array([int(x[col]) for x in r]), np.array([float(x['p_kalibre']) for x in r]),
                np.array([float(x['p_ham']) for x in r]))

    return {
        'ic_test_ASE_LAP': (np.array([int(r['y_true']) for r in it]),
                            np.array([float(r['p_elevated']) for r in it]),
                            np.array([float(r['p_ham']) for r in it])),
        'dis_ASE_LAP': pick('y_ikincil'),
        'dis_Ee_birincil': pick('y_birincil'),
    }


def main():
    L = json.load(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/binary_results.json')))
    thr = L['esik']; lo = L['gri_bolge']['rule_out_alti']; hi = L['gri_bolge']['rule_in_ustu']
    res = {'tohum': SEED, 'bootstrap_tekrar': N_BOOT,
           'kilitli': {'platt': L['kalibrasyon'], 'esik': thr,
                       'gri_bolge': [lo, hi],
                       'kaynak': 'yalniz ic validation seti'},
           'kohortlar': {}}
    for name, (y, p, praw) in cohorts().items():
        rng = np.random.default_rng(SEED)          # her kohort aynı tohumla başlar
        res['kohortlar'][name] = describe(y, p, praw, thr, lo, hi, rng)
        print(f"{name:22s} n={len(y):4d} AUROC={res['kohortlar'][name]['auroc']} "
              f"{res['kohortlar'][name]['auroc_ga']}", flush=True)

    # ön-tanımlı mitral duyarlılık analizi (analysis_plan §5.4): mitral CW / MR / PISA izi çizilen muayeneler dışlanır
    mit = set(json.load(open(os.path.join(HERE, 'mitral_trace_exams.json')))['muayeneler'])
    exd = list(csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv'))))
    res['duyarlilik_mitral'] = {'olcut': 'mitral CW, MR veya PISA izi olan muayeneler dislandi', 'kohortlar': {}}
    for name, col in (('dis_Ee_birincil', 'y_birincil'), ('dis_ASE_LAP', 'y_ikincil')):
        r = [x for x in exd if x[col] != '' and x['af_excluded'] == '0']
        keep = [x for x in r if x['exam_id'] not in mit]
        yk = np.array([int(x[col]) for x in keep]); pk = np.array([float(x['p_kalibre']) for x in keep])
        rng = np.random.default_rng(SEED)
        res['duyarlilik_mitral']['kohortlar'][name] = {
            'dislanan': len(r) - len(keep), 'n': int(len(yk)), 'olay': int(yk.sum()),
            'auroc': round(float(roc_auc_score(yk, pk)), 4), 'auroc_ga': boot_ci(yk, pk, roc_auc_score, rng)}
    print('mitral duyarlılık:', res['duyarlilik_mitral']['kohortlar'], flush=True)

    # ön-tanımlı duyarlılık: üç birincil değişken ölçülmüşse çoğunluk kuralı (>=2/3 anormal)
    ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'echoxflow_reference_v2.csv')))}
    yc, pc = [], []
    for x in exd:
        r = ref.get(x['exam_id'])
        if x['af_excluded'] != '0' or not r or r.get('n_primary_available') != '3' or r.get('n_abnormal') in ('', None):
            continue
        yc.append(int(int(r['n_abnormal']) >= 2)); pc.append(float(x['p_kalibre']))
    yc = np.array(yc); pc = np.array(pc); rng = np.random.default_rng(SEED)
    res['duyarlilik_cogunluk'] = {'n': int(len(yc)), 'olay': int(yc.sum()),
                                  'auroc': round(float(roc_auc_score(yc, pc)), 4),
                                  'auroc_ga': boot_ci(yc, pc, roc_auc_score, rng)}
    print('çoğunluk kuralı:', res['duyarlilik_cogunluk'], flush=True)

    # kırpılma tersilleri (ön-tanımlı sınırlar)
    ex = [x for x in csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv')))
          if x['y_birincil'] != '' and x['af_excluded'] == '0']
    y = np.array([int(r['y_birincil']) for r in ex]); p = np.array([float(r['p_kalibre']) for r in ex])
    cl = np.array([float(r['clip_primary']) if r['clip_primary'] not in ('', 'None') else np.nan for r in ex])
    t = json.load(open(os.path.join(HERE, 'clipping_tertiles.json')))['tersil_sinirlari']
    band = np.digitize(cl, t)
    rng = np.random.default_rng(SEED)
    res['kirpilma'] = {'tersil_sinirlari': t, 'tersiller': []}
    for i in (0, 1, 2):
        s = (band == i) & ~np.isnan(cl)
        res['kirpilma']['tersiller'].append(
            {'tersil': i + 1, 'n': int(s.sum()), 'olay': int(y[s].sum()),
             'auroc': round(float(roc_auc_score(y[s], p[s])), 4),
             'auroc_ga': boot_ci(y[s], p[s], roc_auc_score, rng)})
    s1 = (band == 0) & ~np.isnan(cl); s3 = (band == 2) & ~np.isnan(cl)
    rng = np.random.default_rng(SEED); d = []
    for _ in range(N_BOOT):
        i1 = rng.integers(0, s1.sum(), s1.sum()); i3 = rng.integers(0, s3.sum(), s3.sum())
        a1, b1 = y[s1][i1], p[s1][i1]; a3, b3 = y[s3][i3], p[s3][i3]
        if len(set(a1)) < 2 or len(set(a3)) < 2:
            continue
        d.append(roc_auc_score(a1, b1) - roc_auc_score(a3, b3))
    d = np.array(d)
    res['kirpilma']['T1_eksi_T3'] = {'fark': round(float(d.mean()), 4),
                                     'ga': [round(float(np.percentile(d, 2.5)), 4),
                                            round(float(np.percentile(d, 97.5)), 4)],
                                     'p_fark_sifir_veya_negatif': round(float(np.mean(d <= 0)), 4)}
    # hedef uyumsuzluğu kontrolü: iç testte E/e' hedefi
    sys.path.insert(0, os.path.join(HERE, '..', 'grading'))
    from diastolic_grading import derive, ee_value
    lab = {r['study_id']: r for r in csv.DictReader(open(os.path.expanduser('~/mimic-echo/diastolic_labels.csv')))}
    it = list(csv.DictReader(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/test_predictions.csv'))))
    yy, pp = [], []
    for r in it:
        l = lab.get(r['study_id'])
        if not l:
            continue
        g = lambda k: (float(l[k]) if l[k] not in ('', 'None') else None)
        c = [(g('Ee_sep'), 15), (g('Ee_lat'), 13), (g('Ee_mean'), 14)]
        if any(v is not None for v, _ in c):
            yy.append(int(any(v is not None and v >= t2 for v, t2 in c))); pp.append(float(r['p_elevated']))
    yy = np.array(yy); pp = np.array(pp); rng = np.random.default_rng(SEED)
    res['ic_test_Ee_hedef_uyumsuz'] = {'n': int(len(yy)), 'olay': int(yy.sum()),
                                       'auroc': round(float(roc_auc_score(yy, pp)), 4),
                                       'auroc_ga': boot_ci(yy, pp, roc_auc_score, rng)}
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print(f"\n-> {OUT}")
    print('kırpılma:', [(t['tersil'], t['auroc'], t['auroc_ga']) for t in res['kirpilma']['tersiller']])
    print('T1-T3:', res['kirpilma']['T1_eksi_T3'])
    print('iç test E/e\':', res['ic_test_Ee_hedef_uyumsuz'])


if __name__ == '__main__':
    main()
