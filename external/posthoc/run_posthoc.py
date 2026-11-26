#!/usr/bin/env python3
"""
Birincil sonuçlardan sonraki post hoc analizler — TEK betik, TEK sonuç dosyası.

Bütün analizler POST HOC'tur (analysis_log Ek 12). Kilitli model, Platt katsayıları, eşik ve belirsiz bölge
`~/mimic-echo/runs/b2_binary/binary_results.json` dosyasından okunur, burada yeniden seçilmez.

Bölümler
  faz1      LAVI karşılaştırıcısı (karar kapısı; kural Ek 12'de analizden önce yazıldı)
  kohortlar Üç bölgeli sonuç, çalışma noktaları, Wilson GA'ları (A6); dış kılavuz hedefinde ana küme = üç birincil
            değişkeni de ölçülmüş muayeneler (F0.5)
  ee_tanim  E/e′ hedef tanımı: yalnız ortalama ≥14 (A5)
  karar     Karar eğrisi, p_t %2–10 birincil aralık; net sevk azalması açık formülle (A7, F0.1)
  akis      Dış kohort akışı ve dahil edilen / edilmeyen karşılaştırması (A8)
  ic_etiket İç kohortta etiketin kaç birincil değişkenle verildiği (F0.5'in iç karşılığı), prevalans (F0.3)
  panecho_la  PanEcho'nun hazır LA hacmi / LA boyutu çıktıları karşılaştırıcı olarak (A3)
  ablasyon  Yardımcı başlık ablasyonu, 3 koşul × 3 tohum (A2; koşular bittiyse)
  a4_kohort_disi  Analiz kohortunun dışındaki çalışmalarda model çıktısının dağılımı (A4, Ek 16)
  ek29      Cinsiyet ve yaş alt grupları; hasta düzeyinde (kümelenmiş) bootstrap (Ek 29, iç test seti)

BİRİNCİL İÇ ETİKET (Ek 14, 2026-11-25): sınır düzeltmeli etiket (fix=True). Özgün etiketle sonuçlar `ozgun_etiket`
altında duyarlılık analizi olarak durur. Faz 1 karar kuralı özgün etiketle koşulmuştu (`ozgun_etiket.faz1`).

Çalıştırma (proje kökünden):  python3 external/posthoc/run_posthoc.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, '..')
sys.path.insert(0, EXT); sys.path.insert(0, os.path.join(EXT, '..', 'grading'))
import diastolic_grading as G                                    # noqa: E402
from canonical_results import (SEED, N_BOOT, logit, cal_slope, citl, boot_ci)   # noqa: E402

H = os.path.expanduser('~/mimic-echo/')
OUT = os.path.join(HERE, 'posthoc_results.json')
R4 = lambda x: None if x is None else round(float(x), 4)


# ------------------------------------------------------------------ yardımcılar
def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def prop(k, n):
    return {'k': int(k), 'n': int(n), 'oran': R4(k / n) if n else None, 'ga': wilson(k, n)}


def auc_ci(y, s):
    return {'auroc': R4(roc_auc_score(y, s)), 'ga': boot_ci(y, s, roc_auc_score, np.random.default_rng(SEED))}


def paired_delta(y, a, b):
    """AUROC(a) − AUROC(b), eşleştirilmiş bootstrap."""
    rng = np.random.default_rng(SEED); d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return {'fark': R4(roc_auc_score(y, a) - roc_auc_score(y, b)),
            'ga': [R4(np.percentile(d, 2.5)), R4(np.percentile(d, 97.5))],
            'p_fark_sifir_veya_negatif': R4(np.mean(d <= 0))}


def binary(y, pos):
    """pos: pozitif (sevk) maskesi. Bütün oranlar Wilson GA ile."""
    y = np.asarray(y).astype(bool); pos = np.asarray(pos).astype(bool)
    tp = int((pos & y).sum()); fp = int((pos & ~y).sum()); tn = int((~pos & ~y).sum()); fn = int((~pos & y).sum())
    n = len(y)
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'duyarlilik': prop(tp, tp + fn), 'ozgulluk': prop(tn, tn + fp),
            'PPD': prop(tp, tp + fp), 'NPD': prop(tn, tn + fn),
            'dislanan': prop(tn + fn, n), 'kacirilan_olay': f'{fn}/{tp + fn}'}


def val_threshold(yv, sv, sens):
    """Validasyonda duyarlılığı >= sens veren EN YÜKSEK eşik (pozitif: skor >= eşik). M0'ın dışlama sınırıyla
    aynı kural (train_binary.gray_zone)."""
    fpr, tpr, thr = roc_curve(yv, sv); ok = np.isfinite(thr)
    return float(thr[ok][tpr[ok] >= sens].max())


def matched_threshold(y, s, s0):
    """Duyarlılığı >= s0 olan en yüksek eşik (pozitif: skor >= eşik)."""
    best = None
    for t in np.unique(s):
        if (s[y == 1] >= t).mean() >= s0 - 1e-12:
            best = t
    return float(best)


def three_zone(y, p, lo, hi):
    y = np.asarray(y).astype(bool); n = len(y)
    out_ = p <= lo; in_ = p >= hi; mid = ~out_ & ~in_
    return {'n': int(n), 'olay': int(y.sum()),
            'dislanan': prop(out_.sum(), n), 'NPD': prop((out_ & ~y).sum(), out_.sum()),
            'dislananlarda_kacirilan_olay': f'{int((out_ & y).sum())}/{int(y.sum())}',
            'dahil_edilen': prop(in_.sum(), n), 'PPD': prop((in_ & y).sum(), in_.sum()),
            'belirsiz': prop(mid.sum(), n), 'belirsizde_olay': int((mid & y).sum())}


def describe(y, p, L):
    y = np.asarray(y); p = np.asarray(p)
    d = {'n': int(len(y)), 'olay': int(y.sum()), 'prevalans': R4(y.mean()), **auc_ci(y, p),
         'kalibrasyon_egimi': R4(cal_slope(y, p)),
         'kalibrasyon_egimi_ga': boot_ci(y, p, cal_slope, np.random.default_rng(SEED)),
         'citl': R4(citl(y, p)), 'citl_ga': boot_ci(y, p, citl, np.random.default_rng(SEED)),
         'uc_bolge': three_zone(y, p, L['lo'], L['hi']),
         'nokta_dislama_siniri_val90': binary(y, p > L['lo']),          # birincil çalışma noktası (A6)
         'nokta_youden': binary(y, p >= L['thr']),                       # önceki birincil; artık ikincil
         'nokta_val95_posthoc': binary(y, p >= L['thr95'])}
    return d


def dca(y, p, L):
    """Net fayda = TP/n − (FP/n)·p_t/(1−p_t). 'Sevk' = tam Doppler protokolü.
    (a) standart eğri: sevk <=> kalibre olasılık >= p_t;  (b) kilitli dışlama kuralı (sevk <=> p > lo), her p_t'de.
    Net sevk azalması (1.000 çalışmada) = 1000 × [TN/n − (FN/n) × (1 − p_t)/p_t]  (herkesi sevk etmeye göre)."""
    y = np.asarray(y).astype(bool); n = len(y); prev = y.mean()
    grid = np.round(np.arange(0.02, 0.701, 0.01), 2)

    def nb(pos, pt):
        return ((pos & y).sum() - (pos & ~y).sum() * pt / (1 - pt)) / n

    def reduction(pos, pt):
        tn = (~pos & ~y).sum(); fn = (~pos & y).sum()
        return 1000 * (tn / n - (fn / n) * (1 - pt) / pt)

    rows = []
    for pt in grid:
        rows.append({'pt': float(pt), 'model': R4(nb(p >= pt, pt)), 'kilitli_kural': R4(nb(p > L['lo'], pt)),
                     'herkes': R4(prev - (1 - prev) * pt / (1 - pt)), 'hickimse': 0.0})

    def rng_better(key, lo_, hi_):
        ok = [r['pt'] for r in rows if lo_ <= r['pt'] <= hi_ and r[key] > max(r['herkes'], 0.0) + 1e-9]
        tot = [r['pt'] for r in rows if lo_ <= r['pt'] <= hi_]
        return {'ustun_oldugu_pt_sayisi': len(ok), 'toplam_pt': len(tot),
                'ustun_oldugu_en_dusuk_pt': min(ok) if ok else None}

    out = {'egri': rows,
           'birincil_aralik_2_10': {'model': rng_better('model', 0.02, 0.10),
                                    'kilitli_kural': rng_better('kilitli_kural', 0.02, 0.10)},
           'ek_aralik_2_70': {'model': rng_better('model', 0.02, 0.70),
                              'kilitli_kural': rng_better('kilitli_kural', 0.02, 0.70)},
           'net_sevk_azalmasi_1000': {
               'formul': '1000*[TN/n - (FN/n)*(1-pt)/pt]',
               'kilitli_dislama_siniri': {f'pt_{pt:.4f}': round(float(reduction(p > L['lo'], pt)), 1)
                                          for pt in (0.02, 0.05, 0.10, L['lo'])},
               'youden_esigi_pt_0.1286': round(float(reduction(p >= L['thr'], L['thr'])), 1)}}
    return out


def fnum(v):
    return None if v in ('', 'None', None) else float(v)


def med_iqr(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'medyan': R4(np.median(xs)), 'q1': R4(np.percentile(xs, 25)), 'q3': R4(np.percentile(xs, 75))}


# ------------------------------------------------------------------ veri
def load_internal(fix=False):
    """fix=True: eşik karşılaştırmalarından önce türetilmiş değerler 6 ondalığa yuvarlanır (birim dönüşümünden gelen
    kayan nokta sapması sınır değerlerinde etiketi değiştiriyordu; analysis_log Ek 13)."""
    lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
    split = {r['study_id']: r['split'] for r in csv.DictReader(open(H + 'splits_binary.csv'))}
    pred = {}; order = {}
    for sp in ('val', 'test'):
        for i, r in enumerate(csv.DictReader(open(H + f'runs/b2_binary/{sp}_predictions.csv'))):
            pred[r['study_id']] = (int(r['y_true']), float(r['p_elevated'])); order[r['study_id']] = i
    want = {lab[s]['measurement_id']: s for s in split}
    sm = {k: v for k, v in G.load_study_list().items() if k in want}
    data = G.build_params(sm)
    rows = {}
    for mid, sid in want.items():
        l = lab[sid]; age = fnum(l['age'])
        q = G.derive(dict(data[mid]['params']))
        if fix:
            q = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in q.items()}
        g = G.grade_ase2025(q, age=age)
        e_av = G.eprime_reduced_2025(q, age)[1]; ee_av = G.ee_increased_2025(q)[1]; tr_av = G.tr_increased_2025(q)[1]
        ee = {k: G.ee_value(q, k) for k in ('sep', 'lat', 'mean')}
        rows[sid] = {'split': split[sid], 'lap': g.get('lap'), 'via_sec': 'ikincil_kutu' in g,
                     'n_avail': int(e_av) + int(ee_av) + int(tr_av), 'tr_av': tr_av, 'e_av': e_av, 'ee_av': ee_av,
                     'LAVi': fnum(l['LAVi']), 'LVMi': fnum(l['LVMi']), 'age': age,
                     'female': 1.0 if str(l['gender']).upper().startswith('F') else 0.0,
                     'lap_csv': l['lap_2025'], 'ee': ee, 'prim': g.get('primary'),
                     'y_csv': pred[sid][0] if sid in pred else None, 'p': pred[sid][1] if sid in pred else None,
                     'ord': order.get(sid, 0), 'sid': sid, 'subj': l['subject_id']}
        # y: bu yüklemedeki etiket (fix=False iken tahmin dosyasındaki y_true ile birebir aynı olmalı)
        rows[sid]['y'] = None if g.get('lap') is None else int(g['lap'] == 'elevated')
    return rows


def pick(I, sp):
    """Tahmini ve etiketi olan çalışmalar, tahmin dosyasındaki sırayla (bootstrap dizisi canonical ile aynı kalsın)."""
    return sorted([r for r in I.values() if r['split'] == sp and r['p'] is not None and r['y'] is not None],
                  key=lambda r: r['ord'])


def load_external():
    ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(EXT, 'echoxflow_reference_v2.csv')))}
    pred = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(EXT, 'external_predictions.csv')))}
    rows = {}
    order = {e: i for i, e in enumerate(pred)}
    for e, r in sorted(ref.items(), key=lambda kv: order.get(kv[0], 10 ** 9)):
        p = pred.get(e)
        has_e = bool(r['e_sep'] or r['e_lat']); has_ee = bool(r['Ee_sep'] or r['Ee_lat'] or r['Ee_mean'])
        has_tr = bool(r['TRvel'])
        rows[e] = {'eid': e, 'af': r['af_excluded'] == '1', 'has_e': has_e, 'has_ee': has_ee, 'has_tr': has_tr,
                   'has_E': bool(r['E']), 'n_avail': int(has_e) + int(has_ee) + int(has_tr),
                   'Ee_sep': fnum(r['Ee_sep']), 'Ee_lat': fnum(r['Ee_lat']), 'Ee_mean': fnum(r['Ee_mean']),
                   'e_avg': fnum(r['e_avg']), 'TRvel': fnum(r['TRvel']), 'E': fnum(r['E']), 'EA': fnum(r['EA']),
                   'n_bmode': fnum(r['n_bmode_recordings']), 'fps': fnum(r['bmode_fps']),
                   'n_abn': r['n_abnormal'], 'clip': fnum(r['clip_primary']), 'n_clips': fnum(r['n_target_clips']),
                   'pred': p is not None,
                   'y_ee': (int(p['y_birincil']) if p and p['y_birincil'] != '' else None),
                   'y_lap': (int(p['y_ikincil']) if p and p['y_ikincil'] != '' else None),
                   'p': float(p['p_kalibre']) if p else None}
    return rows


# ------------------------------------------------------------------ A4: analiz kohortunun dışındaki çalışmalar
def a4_outside(IF, L):
    """analysis_log Ek 16. Dışlanan gruplarda doğruluk iddiası yok: yalnız olasılık dağılımı ve bölge oranları.
    Tek ölçüt eşiklerine karşı AUROC'lar referans standart değildir."""
    f = H + 'rev_a4/predictions.csv'
    if not os.path.exists(f):
        return None
    from scipy.stats import spearmanr
    pr = {r['study_id']: (float(r['p_kalibre']), int(r['n_clips'])) for r in csv.DictReader(open(f))}
    man = {r['study_id'] for r in csv.DictReader(open(H + 'manifest_study.csv'))}
    rows = [r for r in csv.DictReader(open(H + 'diastolic_labels.csv')) if r['study_id'] not in man]

    def grp(rs):
        w = [pr[r['study_id']] for r in rs if r['study_id'] in pr]; p = np.array([a for a, _ in w])
        d = {'n': len(rs), 'tahmin_var': prop(len(w), len(rs))}
        if len(w):
            d.update({'klip_sayisi': med_iqr([c for _, c in w]), 'olasilik': med_iqr(list(p)),
                      'dislama_bolgesi': prop((p <= L['lo']).sum(), len(p)),
                      'belirsiz_bolge': prop(((p > L['lo']) & (p < L['hi'])).sum(), len(p)),
                      'dahil_etme_bolgesi': prop((p >= L['hi']).sum(), len(p))})
        return d

    def zones(p):
        p = np.array(p)
        return {'n': len(p), 'olasilik': med_iqr(list(p)), 'dislama_bolgesi': prop((p <= L['lo']).sum(), len(p)),
                'belirsiz_bolge': prop(((p > L['lo']) & (p < L['hi'])).sum(), len(p)),
                'dahil_etme_bolgesi': prop((p >= L['hi']).sum(), len(p))}

    exc = [r for r in rows if r['excluded'] == '1']
    und = [r for r in rows if r['excluded'] == '0' and r['lap_2025'] == '']
    ea44 = [r for r in rows if r['excluded'] == '0' and r['lap_2025'] == 'elevated']
    assert len(exc) + len(und) + len(ea44) == len(rows)
    has = lambda r, t: t in r['excl_reason']
    te = [r for r in IF.values() if r['split'] == 'test' and r['p'] is not None and r['y'] is not None]
    out = {'aciklama': 'analiz manifestinde olmayan, yapılandırılmış ölçümü olan TTE çalışmaları; işleme analiz kohortuyla aynı',
           'hedef_calisma': len(rows), 'tahmin_uretilen_calisma': sum(r['study_id'] in pr for r in rows),
           'islem': {'dicom_112': json.load(open(H + 'rev_a4/prep112_stat.json')),
                     'hires': json.load(open(H + 'rev_a4/hires_stat.json'))},
           'karsilastirma_ic_test': {'tum': zones([r['p'] for r in te]),
                                     'normal_etiket': zones([r['p'] for r in te if r['y'] == 0]),
                                     'yuksek_etiket': zones([r['p'] for r in te if r['y'] == 1])},
           'dislanan': {
               'tum': grp(exc),
               'AF_herhangi': grp([r for r in exc if r['af'] == '1']),
               'AF_EKG_ile': grp([r for r in exc if r['rhythm'] == 'AF']),
               'AF_yalniz_ICD_kodu': grp([r for r in exc if r['rhythm'] == 'possible_AF_icd']),
               'yalniz_AF_baska_neden_yok': grp([r for r in exc if r['excl_reason'] == 'AF']),
               'MR_3_ve_ustu': grp([r for r in exc if has(r, 'MR:')]),
               'mitral_darlik': grp([r for r in exc if has(r, 'MS:')]),
               'MAC_orta_agir': grp([r for r in exc if has(r, 'MAC:')]),
               'AF_disi_nedenler': grp([r for r in exc if r['af'] == '0']),
               'birden_fazla_neden': grp([r for r in exc if ';' in r['excl_reason']])},
           'karar_verilemeyen': {
               'tum': grp(und),
               'birincil_degisken_yetersiz': grp([r for r in und if r['reason_2025'].startswith('primer')]),
               'ikincil_kutuda_LAVI_yok': grp([r for r in und if r['reason_2025'].startswith('ikincil')])}}

    # karar verilemeyenlerde tek ölçütlerle ilişki (referans standart değil)
    U = [(pr[r['study_id']][0], r) for r in und if r['study_id'] in pr]
    rd = lambda r, k: None if r[k] == '' else round(float(r[k]), 6)
    def ee_any(r):
        v = [(rd(r, 'Ee_sep'), 15), (rd(r, 'Ee_lat'), 13), (rd(r, 'Ee_mean'), 14)]
        v = [(x, t) for x, t in v if x is not None]
        return None if not v else int(any(x >= t for x, t in v))
    def rel(val, flag):
        a = [(p, val(r), flag(r)) for p, r in U if val(r) is not None]
        d = {'n': len(a)}
        if len(a) > 10:
            p = np.array([x[0] for x in a]); v = np.array([x[1] for x in a]); y = np.array([x[2] for x in a])
            rng = np.random.default_rng(SEED); bs = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(p), len(p)); bs.append(spearmanr(p[i], v[i])[0])
            d.update({'spearman': R4(spearmanr(p, v)[0]), 'spearman_ga': [R4(np.percentile(bs, 2.5)), R4(np.percentile(bs, 97.5))],
                      'olcut_pozitif': int(y.sum())})
            if 0 < y.sum() < len(y):
                d['olcute_karsi'] = auc_ci(y, p)
        return d
    out['karar_verilemeyen']['tek_olcutlerle_iliski'] = {
        'not': 'tek ölçüt eşikleri referans standart değildir',
        'LAVI_gt34': rel(lambda r: rd(r, 'LAVi'), lambda r: int(rd(r, 'LAVi') > 34)),
        'Ee_ortalama__herhangi_biri_yuksek': rel(lambda r: rd(r, 'Ee_mean'), ee_any),
        'Ee_herhangi_biri_tum_olculenler': (lambda a: {'n': len(a), 'olcut_pozitif': int(sum(y for _, y in a)),
                                                      **({'olcute_karsi': auc_ci(np.array([y for _, y in a]), np.array([p for p, _ in a]))} if a else {})})(
            [(p, ee_any(r)) for p, r in U if ee_any(r) is not None]),
        'TR_ge2_8': rel(lambda r: rd(r, 'TRvel'), lambda r: int(rd(r, 'TRvel') >= 2.8))}

    # E/A eksik, kılavuz etiketi "yüksek" (44): dışlama sınırında işaretlenen
    w = np.array([pr[r['study_id']][0] for r in ea44 if r['study_id'] in pr])
    out['EA_eksik_LAP_yuksek'] = {**grp(ea44), 'dislama_sinirinda_isaretlenen': prop((w > L['lo']).sum(), len(w)),
                                  'youden_esiginde_isaretlenen': prop((w >= L['thr']).sum(), len(w))}
    return out


# ------------------------------------------------------------------ Faz 1
def faz1(I, L):
    tr = [r for r in I.values() if r['split'] == 'train' and r['y'] is not None]
    va = pick(I, 'val'); te = pick(I, 'test')
    ylab = lambda r: r['y']
    cols = ('LVMi', 'age', 'female', 'LAVi')
    cc = lambda r: all(r[c] is not None for c in cols)

    # M3: yalnız eğitim seti (tam-olgu)
    trc = [r for r in tr if cc(r)]
    Xtr = np.array([[r[c] for c in cols] for r in trc]); mu = Xtr.mean(0); sd = Xtr.std(0)
    z = lambda rs: (np.array([[r[c] for c in cols] for r in rs]) - mu) / sd
    m3 = LogisticRegression(C=1e6, max_iter=5000).fit(z(trc), np.array([ylab(r) for r in trc]))
    # M4: validasyon seti (M0'ın eğitim seti tahmini yok) — Ek 12'de belirtilen ayrılma
    vac = [r for r in va if cc(r)]
    m4 = LogisticRegression(C=1e6, max_iter=5000).fit(
        np.column_stack([z(vac), logit([r['p'] for r in vac])]), np.array([r['y'] for r in vac]))

    def score(rs, name):
        if name == 'M0':
            return np.array([r['p'] for r in rs])
        if name == 'M1':
            return np.array([r['LAVi'] for r in rs])
        if name == 'M2':
            return np.array([float(r['LAVi'] > 34.0) for r in rs])
        if name == 'M3':
            return m3.predict_proba(z(rs))[:, 1]
        if name == 'M4':
            return m4.predict_proba(np.column_stack([z(rs), logit([r['p'] for r in rs])]))[:, 1]

    def compare(rs_all, ad):
        res = {'ad': ad, 'n_kume': len(rs_all), 'olay_kume': int(sum(r['y'] for r in rs_all)),
               'LAVI_eksik': int(sum(r['LAVi'] is None for r in rs_all)),
               'LAVI_eksik_olay': int(sum(r['y'] for r in rs_all if r['LAVi'] is None)), 'modeller': {}}
        for name in ('M1', 'M2', 'M3', 'M4'):
            rs = [r for r in rs_all if (cc(r) if name in ('M3', 'M4') else r['LAVi'] is not None)]
            y = np.array([r['y'] for r in rs]); p0 = score(rs, 'M0'); s = score(rs, name)
            s0 = float((p0[y == 1] > L['lo']).mean())
            d = {'n': len(rs), 'olay': int(y.sum()), 'M0': auc_ci(y, p0), 'karsilastirici': auc_ci(y, s),
                 'delta_M0_eksi_karsilastirici': paired_delta(y, p0, s),
                 'M0_dislama_sinirinda': binary(y, p0 > L['lo'])}
            if name == 'M2':
                d['sabit_esikte_LAVI_gt_34'] = binary(y, s > 0.5)
            else:
                t = matched_threshold(y, s, s0)
                d['eslestirilmis_duyarlilikta'] = {'esik': R4(t), **binary(y, s >= t)}
                # dışlanan oran farkı (M0 − karşılaştırıcı), eşik her yeniden örneklemde yeniden eşleştirilir
                rng = np.random.default_rng(SEED); df = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(y), len(y))
                    if len(set(y[i])) < 2:
                        continue
                    sb = float((p0[i][y[i] == 1] > L['lo']).mean())
                    tb = matched_threshold(y[i], s[i], sb)
                    df.append((p0[i] <= L['lo']).mean() - (s[i] < tb).mean())
                d['dislanan_oran_farki_M0_eksi_karsilastirici'] = {
                    'fark': R4((p0 <= L['lo']).mean() - (s < t).mean()),
                    'ga': [R4(np.percentile(df, 2.5)), R4(np.percentile(df, 97.5))]}
                # validasyondan gelen eşik (kural 4 ile uyumlu, betimleyici)
                vs = [r for r in va if (cc(r) if name in ('M3', 'M4') else r['LAVi'] is not None)]
                tv = val_threshold(np.array([r['y'] for r in vs]), score(vs, name), 0.90)
                d['validasyon_esiginde_val90'] = {'esik': R4(tv), 'val_n': len(vs), **binary(y, s >= tv)}
            res['modeller'][name] = d
        return res

    lavi_free = [r for r in te if not r['via_sec']]
    out = {'egitim_tam_olgu_n': len(trc), 'egitim_tam_olgu_olay': int(sum(ylab(r) for r in trc)),
           'M3_katsayilar_z': {c: R4(v) for c, v in zip(cols, m3.coef_[0])},
           'M4_val_n': len(vac), 'M4_katsayilar_z': {c: R4(v) for c, v in zip(cols + ('logit_M0',), m4.coef_[0])},
           'birincil_kume_LAVIsiz_etiket': compare(lavi_free, 'iç test, etiketi ikincil değişkensiz'),
           'ikincil_kume_tum_test': compare(te, 'tüm iç test seti (LAVI etiketin parçası)')}
    b = out['birincil_kume_LAVIsiz_etiket']['modeller']['M1']
    k1 = b['delta_M0_eksi_karsilastirici']['fark'] >= 0
    k2 = b['delta_M0_eksi_karsilastirici']['ga'][0] > -0.05
    k3 = b['dislanan_oran_farki_M0_eksi_karsilastirici']['fark'] >= 0
    out['karar'] = {'kosul_1_delta_nokta_ge_0': bool(k1), 'kosul_2_delta_ga_alt_gt_-0.05': bool(k2),
                    'kosul_3_dislanan_oran_M0_ge_M1': bool(k3),
                    'senaryo': 'A' if (k1 and k2 and k3) else 'B'}
    return out


# ------------------------------------------------------------------ Ek 29: alt gruplar, hasta düzeyinde bootstrap
def cluster_ci(groups, fn):
    """Hasta düzeyinde bootstrap: hastalar yerine koyarak çekilir, çekilen hastanın bütün çalışmaları alınır.
    fn(indeks dizisi) -> istatistik."""
    rng = np.random.default_rng(SEED); groups = np.asarray(groups)
    ids = sorted(set(groups.tolist())); idx = {g: np.where(groups == g)[0] for g in ids}
    v = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
        try:
            v.append(fn(i))
        except ValueError:
            continue
    return [R4(np.percentile(v, 2.5)), R4(np.percentile(v, 97.5))]


def ek29(IF, L):
    te = pick(IF, 'test')
    arr = lambda rs, k: np.array([r[k] for r in rs])
    # --- 29.1 alt gruplar
    def sub(rs):
        y = arr(rs, 'y'); p = arr(rs, 'p')
        return {'n': len(rs), 'olay': int(y.sum()), 'yorumlanamayacak_kadar_kucuk': bool(y.sum() < 10),
                **auc_ci(y, p), 'uc_bolge': three_zone(y, p, L['lo'], L['hi'])}
    def diff(ra, rb):
        ya, pa, yb, pb = arr(ra, 'y'), arr(ra, 'p'), arr(rb, 'y'), arr(rb, 'p')
        rng = np.random.default_rng(SEED); d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(ya), len(ya)); j = rng.integers(0, len(yb), len(yb))
            if len(set(ya[i])) < 2 or len(set(yb[j])) < 2:
                continue
            d.append(roc_auc_score(ya[i], pa[i]) - roc_auc_score(yb[j], pb[j]))
        return {'fark': R4(roc_auc_score(ya, pa) - roc_auc_score(yb, pb)),
                'ga': [R4(np.percentile(d, 2.5)), R4(np.percentile(d, 97.5))]}
    fem = [r for r in te if r['female'] == 1.0]; mal = [r for r in te if r['female'] == 0.0]
    old = [r for r in te if r['age'] is not None and r['age'] >= 65]
    yng = [r for r in te if r['age'] is not None and r['age'] < 65]
    alt = {'kadin': sub(fem), 'erkek': sub(mal), 'kadin_eksi_erkek': diff(fem, mal),
           'yas_ge65': sub(old), 'yas_lt65': sub(yng), 'ge65_eksi_lt65': diff(old, yng),
           'yas_eksik': int(sum(r['age'] is None for r in te))}
    # --- 29.2 hasta düzeyinde bootstrap
    def clus(rs, comp=None):
        y = arr(rs, 'y'); p = arr(rs, 'p'); g = arr(rs, 'subj')
        cnt = {}
        for s_ in g.tolist():
            cnt[s_] = cnt.get(s_, 0) + 1
        d = {'n': len(rs), 'olay': int(y.sum()), 'hasta': len(cnt),
             'birden_cok_calismasi_olan_hasta': int(sum(v > 1 for v in cnt.values())),
             'auroc': R4(roc_auc_score(y, p)), 'ga_calisma_duzeyi': auc_ci(y, p)['ga'],
             'ga_hasta_duzeyi': cluster_ci(g, lambda i: roc_auc_score(y[i], p[i]))}
        if comp is not None:
            s = arr(rs, comp)
            d['LAVI_auroc'] = R4(roc_auc_score(y, s))
            d['delta_M0_eksi_LAVI'] = {
                'fark': R4(roc_auc_score(y, p) - roc_auc_score(y, s)),
                'ga_calisma_duzeyi': paired_delta(y, p, s)['ga'],
                'ga_hasta_duzeyi': cluster_ci(g, lambda i: roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], s[i]))}
        return d
    lf = [r for r in te if not r['via_sec']]
    has = lambda rs: [r for r in rs if r['LAVi'] is not None]
    hd = {'ic_test': clus(te), 'ic_test_LAVIsiz_etiket': clus(lf),
          'LAVI_karsilastirmasi_LAVIsiz_etiket': clus(has(lf), 'LAVi'),
          'LAVI_karsilastirmasi_tum_test': clus(has(te), 'LAVi')}
    return {'aciklama': 'Ek 29 (post hoc): iç test seti, düzeltilmiş etiket, kilitli model ve eşikler',
            'alt_gruplar': alt, 'hasta_duzeyi': hd}


# ------------------------------------------------------------------ A3 / A2 yardımcıları
def compare_score(y, p0, s, L):
    """Karşılaştırıcı skor s (yüksek = daha olası) ile M0: AUROC, eşleştirilmiş fark, eşleştirilmiş duyarlılıkta dışlama."""
    y = np.asarray(y); p0 = np.asarray(p0); s = np.asarray(s)
    s0 = float((p0[y == 1] > L['lo']).mean()); t = matched_threshold(y, s, s0)
    return {'n': int(len(y)), 'olay': int(y.sum()), 'M0': auc_ci(y, p0), 'karsilastirici': auc_ci(y, s),
            'delta_M0_eksi_karsilastirici': paired_delta(y, p0, s),
            'M0_dislama_sinirinda': binary(y, p0 > L['lo']),
            'eslestirilmis_duyarlilikta': {'esik': R4(t), **binary(y, s >= t)}}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return R4(np.corrcoef(ra, rb)[0, 1])


def panecho_la(IF, I, X, L):
    d = os.path.expanduser('~/mimic-echo/runs/rev_panecho_la')
    fx = os.path.join(HERE, 'external_panecho_la.csv')
    if not (os.path.exists(os.path.join(d, 'test_la.csv')) and os.path.exists(fx)):
        return None
    la = {r['study_id']: (float(r['lavol_cm3']), float(r['p_la_dilated'])) for r in csv.DictReader(open(os.path.join(d, 'test_la.csv')))}
    lx = {r['exam_id']: (float(r['lavol_cm3']), float(r['p_la_dilated'])) for r in csv.DictReader(open(fx))}
    out = {'aciklama': 'PanEcho önceden eğitilmiş LAVol (cm³, indekssiz) ve LASize (1 − P(Normal)); ince ayar yok'}
    te = pick(IF, 'test'); m = [r for r in te if r['LAVi'] is not None]
    out['ic_test_olculmus_LAVI_ile_spearman'] = {
        'n': len(m), 'LAVol': spearman([la[r['sid']][0] for r in m], [r['LAVi'] for r in m]),
        'LASize': spearman([la[r['sid']][1] for r in m], [r['LAVi'] for r in m]),
        'M0': spearman([r['p'] for r in m], [r['LAVi'] for r in m])}
    sets = {'ic_test': te, 'ic_test_LAVIsiz_etiket': [r for r in te if not r['via_sec']],
            'ic_test_ozgun_etiket': pick(I, 'test'),
            'ic_test_ozgun_etiket_LAVIsiz': [r for r in pick(I, 'test') if not r['via_sec']]}
    for k, rs in sets.items():
        out[k] = {nm: compare_score([r['y'] for r in rs], [r['p'] for r in rs], [la[r['sid']][j] for r in rs], L)
                  for j, nm in ((0, 'LAVol'), (1, 'LASize'))}
    ex = [r for r in X.values() if not r['af'] and r['pred']]
    xs = {'dis_Ee_birincil': ([r for r in ex if r['y_ee'] is not None], 'y_ee'),
          'dis_kilavuz_uc_olcum_ANA': ([r for r in ex if r['y_lap'] is not None and r['n_avail'] == 3], 'y_lap'),
          'dis_kilavuz_tum_duyarlilik': ([r for r in ex if r['y_lap'] is not None], 'y_lap')}
    for k, (rs, yk) in xs.items():
        out[k] = {nm: compare_score([r[yk] for r in rs], [r['p'] for r in rs], [lx[r['eid']][j] for r in rs], L)
                  for j, nm in ((0, 'LAVol'), (1, 'LASize'))}
    return out


def ablation(IF, I, X):
    rd = os.path.expanduser('~/mimic-echo/runs/rev_abl'); fx = os.path.join(HERE, 'ablation_external.csv')
    runs = sorted(d for d in (os.listdir(rd) if os.path.isdir(rd) else [])
                  if os.path.exists(os.path.join(rd, d, 'binary_results.json')))
    if not runs:
        return None
    px = {}
    if os.path.exists(fx):
        for r in csv.DictReader(open(fx)):
            px.setdefault(r['kosu'], {})[r['exam_id']] = float(r['p_ham'])
    teF = pick(IF, 'test'); teO = pick(I, 'test')
    ex = [r for r in X.values() if not r['af'] and r['pred']]
    sets_x = {'dis_Ee_birincil': ([r for r in ex if r['y_ee'] is not None], 'y_ee'),
              'dis_kilavuz_uc_olcum_ANA': ([r for r in ex if r['y_lap'] is not None and r['n_avail'] == 3], 'y_lap'),
              'dis_kilavuz_tum_duyarlilik': ([r for r in ex if r['y_lap'] is not None], 'y_lap')}
    sets_i = {'ic_test': teF, 'ic_test_LAVIsiz_etiket': [r for r in teF if not r['via_sec']], 'ic_test_ozgun_etiket': teO}

    def aucs(pi, pe):
        o = {k: R4(roc_auc_score([r['y'] for r in rs], [pi[r['sid']] for r in rs])) for k, rs in sets_i.items()}
        if pe:
            o.update({k: R4(roc_auc_score([r[yk] for r in rs], [pe[r['eid']] for r in rs])) for k, (rs, yk) in sets_x.items()})
        return o
    out = {'kosular': {}, 'kilitli_model': aucs({r['sid']: r['p'] for r in teO}, {r['eid']: r['p'] for r in ex})}
    for run in runs:
        b = json.load(open(os.path.join(rd, run, 'binary_results.json')))
        pi = {r['study_id']: float(r['p_ham']) for r in csv.DictReader(open(os.path.join(rd, run, 'test_predictions.csv')))}
        out['kosular'][run] = {'en_iyi_epoch': b['best_epoch'], 'val_auroc': b['val_auroc_best'], **aucs(pi, px.get(run))}
    keys = list(sets_i) + list(sets_x)
    summ = {}
    for cfg in ('K', 'A0', 'A1'):
        rs = [out['kosular'][f'{cfg}_s{sd}'] for sd in (42, 43, 44) if f'{cfg}_s{sd}' in out['kosular']]
        summ[cfg] = {'tohum_sayisi': len(rs),
                     **{k: {'ortalama': R4(np.mean([r[k] for r in rs])), 'min': min(r[k] for r in rs), 'max': max(r[k] for r in rs)}
                        for k in keys if rs and all(k in r for r in rs)}}
    for cfg in ('A0', 'A1'):
        d = {}
        for k in keys:
            v = [out['kosular'][f'{cfg}_s{sd}'][k] - out['kosular'][f'K_s{sd}'][k] for sd in (42, 43, 44)
                 if f'{cfg}_s{sd}' in out['kosular'] and f'K_s{sd}' in out['kosular']
                 and k in out['kosular'][f'{cfg}_s{sd}'] and k in out['kosular'][f'K_s{sd}']]
            if v:
                d[k] = {'ortalama_fark': R4(np.mean(v)), 'min': R4(min(v)), 'max': R4(max(v)), 'tohum_sayisi': len(v)}
        summ[f'{cfg}_eksi_K_ayni_tohum'] = d
    out['ozet'] = summ
    out['tamamlanan_kosu'] = len(out['kosular'])
    out['tamamlandi'] = bool(len(out['kosular']) == 9 and os.path.exists(os.path.join(HERE, 'ablation_external.csv')))
    return out


# ------------------------------------------------------------------ ana
def internal_block(I_, L, verbose):
    """Bir etiket sürümü için iç kohort sonuçları ve Faz 1."""
    te = pick(I_, 'test'); C = {}
    def add(name, rs, acik, y=None):
        yy = np.array([r['y'] for r in rs] if y is None else y); pp = np.array([r['p'] for r in rs])
        C[name] = {'aciklama': acik, **describe(yy, pp, L), 'karar_egrisi': dca(yy, pp, L)}
        if verbose:
            d = C[name]
            print(f"{name:34s} n={d['n']:4d} olay={d['olay']:3d} AUROC={d['auroc']} {d['ga']} "
                  f"dışlanan={d['uc_bolge']['dislanan']['oran']} NPD={d['uc_bolge']['NPD']['oran']} "
                  f"kaçan={d['uc_bolge']['dislananlarda_kacirilan_olay']}", flush=True)
    add('ic_test', te, 'iç test seti, kılavuz hedefi')
    add('ic_test_LAVIsiz_etiket', [r for r in te if not r['via_sec']], 'iç test, etiketi ikincil değişkensiz (uyumlu olgular)')
    add('ic_test_uc_olcum', [r for r in te if r['n_avail'] == 3], 'iç test, üç birincil değişken de ölçülmüş')
    add('ic_test_uc_olcum_LAVIsiz_etiket', [r for r in te if r['n_avail'] == 3 and not r['via_sec']],
        'iç test, üç birincil ölçülmüş ve etiket ikincil değişkensiz (dış ana kümenin benzer-benzere karşılığı)')
    s = [r for r in te if all(r['ee'][k] is not None for k in ('sep', 'lat', 'mean'))]
    add('ic_test_Ee_ortalama_ge14', s, 'iç test, yalnız ortalama E/e′ ≥14; iki bölge de ölçülmüş', y=[int(r['ee']['mean'] >= 14) for r in s])
    s = [r for r in te if r['ee_av']]
    add('ic_test_Ee_herhangi', s, 'iç test, yüksek E/e′ (herhangi biri)',
        y=[int(any(r['ee'][k] is not None and r['ee'][k] >= t for k, t in (('sep', 15), ('lat', 13), ('mean', 14)))) for r in s])
    allr = [r for r in I_.values() if r['lap'] is not None]
    def avail_tab(rs):
        t = {}
        for r in rs:
            k = f"{r['n_avail']}_degisken|{'TR_var' if r['tr_av'] else 'TR_yok'}|{'ikincil_kutu' if r['via_sec'] else 'birincil'}|{r['lap']}"
            t[k] = t.get(k, 0) + 1
        return dict(sorted(t.items()))
    lab = {'n': len(allr), 'yuksek_LAP': int(sum(r['lap'] == 'elevated' for r in allr)),
           'prevalans': R4(np.mean([r['lap'] == 'elevated' for r in allr])),
           'bolunme': {sp: {'n': sum(r['split'] == sp for r in allr),
                            'yuksek_LAP': int(sum(r['split'] == sp and r['lap'] == 'elevated' for r in allr))}
                       for sp in ('train', 'val', 'test')},
           'TR_olculmus': prop(sum(r['tr_av'] for r in allr), len(allr)),
           'uc_birincil_de_olculmus': prop(sum(r['n_avail'] == 3 for r in allr), len(allr)),
           'ikincil_kutuyla_etiketlenen': prop(sum(r['via_sec'] for r in allr), len(allr)),
           'dagilim_tum': avail_tab(allr), 'dagilim_test': avail_tab(te)}
    return {'ic_etiket': lab, 'kohortlar': C, 'faz1': faz1(I_, L)}


def main():
    B = json.load(open(H + 'runs/b2_binary/binary_results.json'))
    L = {'thr': B['esik'], 'lo': B['gri_bolge']['rule_out_alti'], 'hi': B['gri_bolge']['rule_in_ustu']}
    I = load_internal(fix=False); IF = load_internal(fix=True); X = load_external()
    va = pick(I, 'val'); te = pick(I, 'test')
    assert all(r['y'] == r['y_csv'] for r in va + te) and len(te) == 438, 'özgün etiket yeniden üretimi tahmin dosyasıyla uyuşmuyor'
    yv = np.array([r['y'] for r in va]); pv = np.array([r['p'] for r in va])
    L['thr95'] = val_threshold(yv, pv, 0.95)          # özgün validasyon etiketleri: kilitli eşiklerle aynı kaynak
    res = {'uretim': 'external/posthoc/run_posthoc.py', 'tohum': SEED, 'bootstrap_tekrar': N_BOOT,
           'not': 'Bütün analizler POST HOC (analysis_log Ek 12–14). Birincil iç etiket: sınır düzeltmeli.',
           'kilitli': {'platt': B['kalibrasyon'], 'youden_esigi': L['thr'], 'dislama_siniri_val90': L['lo'],
                       'dahil_etme_siniri_val90': L['hi'], 'val95_esigi_posthoc': R4(L['thr95']),
                       'val_kontrol_ozgun_etiket': {'dislama_sinirinda_duyarlilik': R4((pv[yv == 1] > L['lo']).mean()),
                                                    'val95_esiginde_duyarlilik': R4((pv[yv == 1] >= L['thr95']).mean())}}}
    vaF = pick(IF, 'val'); yvF = np.array([r['y'] for r in vaF]); pvF = np.array([r['p'] for r in vaF])
    res['kilitli']['val_kontrol_duzeltilmis_etiket'] = {
        'n': len(vaF), 'olay': int(yvF.sum()), 'dislama_sinirinda_duyarlilik': R4((pvF[yvF == 1] > L['lo']).mean()),
        'val95_esiginde_duyarlilik': R4((pvF[yvF == 1] >= L['thr95']).mean())}

    # --- etiket düzeltmesi (Ek 13)
    ch = {}
    for sid, r in I.items():
        a, b = r['lap'], IF[sid]['lap']
        if a != b:
            k = f"{r['split']}|{a}->{b}"; ch[k] = ch.get(k, 0) + 1
    both = [s for s in I if I[s]['prim'] is not None and IF[s]['prim'] is not None]
    thr_ext = (('e_avg', 6.5), ('Ee_sep', 15), ('Ee_lat', 13), ('Ee_mean', 14), ('TRvel', 2.8), ('EA', 0.8))
    res['etiket_sinir_duzeltmesi'] = {
        'aciklama': 'türetilmiş değerler eşik karşılaştırmasından önce 6 ondalığa yuvarlandı; model, kalibratör, eşikler aynı',
        'degisen_etiket': dict(sorted(ch.items())), 'degisen_toplam': int(sum(ch.values())),
        'e_azalmis_bayragi_degisen': int(sum(I[s]['prim']['e_reduced'] != IF[s]['prim']['e_reduced'] for s in both)),
        'Ee_bayragi_degisen': int(sum(I[s]['prim']['ee_high'] != IF[s]['prim']['ee_high'] for s in both)),
        'egitim_olayi_ozgun': int(sum(r['split'] == 'train' and r['lap'] == 'elevated' for r in I.values())),
        'dis_kohortta_sinirda_deger': int(sum(any(r[k] is not None and abs(r[k] - t) < 1e-6 for k, t in thr_ext) for r in X.values()))}

    # --- iç kohort: birincil (düzeltilmiş) ve özgün etiket
    print('--- birincil: düzeltilmiş etiket', flush=True)
    P = internal_block(IF, L, True)
    res['ic_etiket'] = P['ic_etiket']; res['faz1'] = P['faz1']; C = P['kohortlar']
    print('--- duyarlılık: özgün etiket', flush=True)
    res['ozgun_etiket'] = internal_block(I, L, True)
    res['ozgun_etiket']['not'] = 'Faz 1 karar kuralı (Ek 12) bu etiketle koşuldu.'

    # --- dış kohortlar (A6, A9, F0.5, A5)
    def ext(sel):
        return [r for r in X.values() if not r['af'] and r['pred'] and sel(r)]
    def add(name, y, p, acik):
        C[name] = {'aciklama': acik, **describe(np.array(y), np.array(p), L), 'karar_egrisi': dca(np.array(y), np.array(p), L)}
        d = C[name]
        print(f"{name:34s} n={d['n']:4d} olay={d['olay']:3d} AUROC={d['auroc']} {d['ga']} "
              f"dışlanan={d['uc_bolge']['dislanan']['oran']} NPD={d['uc_bolge']['NPD']['oran']} "
              f"kaçan={d['uc_bolge']['dislananlarda_kacirilan_olay']}", flush=True)
    s = ext(lambda r: r['y_lap'] is not None and r['n_avail'] == 3)
    add('dis_kilavuz_uc_olcum_ANA', [r['y_lap'] for r in s], [r['p'] for r in s],
        'dış, kılavuz hedefi, üç birincil değişken de ölçülmüş (ana küme)')
    s = ext(lambda r: r['y_lap'] is not None)
    add('dis_kilavuz_tum_duyarlilik', [r['y_lap'] for r in s], [r['p'] for r in s],
        'dış, kılavuz hedefi, iki ya da üç birincil değişkenle karar (önceki 191; duyarlılık analizi)')
    res['dis_kilavuz_eksik_degisken'] = {
        'toplam': len(s), 'uc_olcum': sum(r['n_avail'] == 3 for r in s),
        'TR_yok': sum(not r['has_tr'] for r in s), 'Ee_yok': sum(not r['has_ee'] for r in s),
        'TR_yok_olay': sum(r['y_lap'] for r in s if not r['has_tr']),
        'Ee_yok_olay': sum(r['y_lap'] for r in s if not r['has_ee'])}
    _m = [r for r in s if r['n_avail'] == 3]
    res['dis_kilavuz_eksik_degisken']['ANA_anormal_sayisi_x_etiket'] = {
        k: sum(f"{r['n_abn']}|{r['y_lap']}" == k for r in _m) for k in sorted({f"{r['n_abn']}|{r['y_lap']}" for r in _m})}
    s = ext(lambda r: r['y_ee'] is not None)
    add('dis_Ee_birincil', [r['y_ee'] for r in s], [r['p'] for r in s], "dış, yüksek E/e′ (ön-tanımlı birincil; herhangi biri)")
    s = ext(lambda r: r['Ee_sep'] is not None and r['Ee_lat'] is not None and r['Ee_mean'] is not None)
    add('dis_Ee_ortalama_ge14', [int(r['Ee_mean'] >= 14) for r in s], [r['p'] for r in s],
        'dış, yalnız ortalama E/e′ ≥14; septal ve lateral birlikte ölçülmüş')
    C['dis_Ee_ortalama_ge14']['herhangi_biri_tanimiyla_ayni_kume'] = auc_ci(
        np.array([r['y_ee'] for r in s]), np.array([r['p'] for r in s]))
    C['dis_Ee_ortalama_ge14']['herhangi_biri_olay'] = int(sum(r['y_ee'] for r in s))
    res['kohortlar'] = C

    # --- A8: dış akış ve seçilim
    nonaf = [r for r in X.values() if not r['af']]
    in303 = [r for r in nonaf if r['y_ee'] is not None]; out303 = [r for r in nonaf if r['y_ee'] is None]
    g191 = [r for r in nonaf if r['y_lap'] is not None]
    def desc(rs):
        return {'n': len(rs), 'bmode_kayit': med_iqr([r['n_bmode'] for r in rs]), 'kare_hizi': med_iqr([r['fps'] for r in rs]),
                'TR_olculmus': prop(sum(r['has_tr'] for r in rs), len(rs)), 'TR_hizi': med_iqr([r['TRvel'] for r in rs]),
                'E_olculmus': prop(sum(r['has_E'] for r in rs), len(rs)), 'E_cms': med_iqr([r['E'] for r in rs]),
                'eprime_olculmus': prop(sum(r['has_e'] for r in rs), len(rs)),
                'E_e_ort': med_iqr([r['Ee_mean'] for r in rs]),
                'kirpilma_skoru': med_iqr([r['clip'] for r in rs]),
                'model_olasiligi': med_iqr([r['p'] for r in rs])}
    notin = [r for r in in303 if r['y_lap'] is None]
    res['akis_dis'] = {
        'toplam_muayene': len(X), 'ritim_dislamasi': sum(r['af'] for r in X.values()), 'ritim_sonrasi': len(nonaf),
        'Ee_hesaplanamayan': {
            'toplam': len(out303),
            'E_ve_eprime_yok': sum((not r['has_E']) and (not r['has_e']) for r in out303),
            'yalniz_E_var_eprime_yok': sum(r['has_E'] and not r['has_e'] for r in out303),
            'yalniz_eprime_var_E_yok': sum((not r['has_E']) and r['has_e'] for r in out303)},
        'Ee_degerlendirilebilir': len(in303),
        'kilavuz_hedefi_karar_verilemeyen_303_icinde': {
            'toplam': len(notin),
            'anormal_sayisi_dagilimi': {k: sum(r['n_abn'] == k for r in notin) for k in sorted({r['n_abn'] for r in notin})},
            'TR_yok': sum(not r['has_tr'] for r in notin)},
        'kilavuz_hedefi_karar_verilen': {'toplam': len(g191), 'Ee_kumesinin_disindan': sum(r['y_ee'] is None for r in g191)},
        'kilavuz_hedefi_uc_olcum_ANA': sum(r['n_avail'] == 3 for r in g191),
        'karsilastirma': {'Ee_kumesinde': desc(in303), 'Ee_kumesi_disinda_ritim_ok': desc(out303),
                          'kilavuz_ANA_117': desc([r for r in g191 if r['n_avail'] == 3]),
                          '303_icinde_ANA_disinda': desc([r for r in in303 if not (r['y_lap'] is not None and r['n_avail'] == 3)])}}

    # --- A3, A2
    pl = panecho_la(IF, I, X, L)
    if pl:
        res['panecho_la'] = pl
    ab = ablation(IF, I, X)
    if ab:
        res['ablasyon'] = ab
    a4 = a4_outside(IF, L)
    if a4:
        res['a4_kohort_disi'] = a4

    res['ek29'] = ek29(IF, L)
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print('\n-> ' + OUT)
    print('FAZ 1 (birincil, düzeltilmiş etiket):', res['faz1']['karar'])
    print('FAZ 1 (ön-tanımlı koşu, özgün etiket):', res['ozgun_etiket']['faz1']['karar'])


if __name__ == '__main__':
    main()
