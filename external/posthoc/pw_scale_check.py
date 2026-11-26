#!/usr/bin/env python3
"""PW kaliper ölçeğinin saklanan spektruma göre ölçümü ve duyarlılık analizi (analysis_log Ek 26; post hoc).

  python3 external/posthoc/pw_scale_check.py measure      -> pw_scale_calipers.csv, pw_scale_results.json['olcum']
  python3 external/posthoc/pw_scale_check.py scenarios    -> pw_scale_results.json['senaryolar']

Referans dosyaları, model, kalibrasyon ve eşikler değişmez. Çalıştırma: proje kökünden."""
from __future__ import annotations
import csv, glob, json, os, shutil, sys, tarfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(ROOT, 'grading'))
sys.path.insert(0, os.path.join(ROOT, 'echoxflow-code', 'src'))
import run_posthoc as RR                                              # noqa: E402
from build_external_reference import corrected                         # noqa: E402
from diastolic_grading import derive, grade_ase2025, ee_value          # noqa: E402

SEED, N_BOOT = 20260901, 4000
OUT_JSON, OUT_CSV = os.path.join(HERE, 'pw_scale_results.json'), os.path.join(HERE, 'pw_scale_calipers.csv')
TMP = os.path.expanduser('~/echoxflow-render/_pw')
US = np.linspace(0.3, 3.0, 271)
LAB = {'MV E/A Velocity': 'MV', 'MV Eprime Septal Velocity': 'e_sep', 'MV Eprime Lateral Velocity': 'e_lat', 'TR Vmax': 'TR'}
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)


def lab_of(label):
    for k, v in LAB.items():
        if (label or '').endswith(k):
            return v
    return None


# ------------------------------------------------------------------ kenar kestiricileri (Ek 26)
def sample(prof, zero, sign, d):
    N = len(prof); rows = zero - sign * US * d
    ok = (rows >= 1) & (rows <= N - 2)
    vals = np.full(len(US), np.nan); vals[ok] = np.interp(rows[ok], np.arange(N), prof)
    return vals, ok


def tail_of(vals, ok):
    v = vals[ok]
    return float(np.median(v[-max(8, int(0.12 * len(v))):]))


def edge_plateau(vals, ok, level):
    if ok.sum() < 60:
        return None
    seg = vals[(US > 0.5) & (US < 0.8)]
    if not np.isfinite(seg).any():
        return None
    pl, tail = float(np.nanmedian(seg)), tail_of(vals, ok)
    if pl - tail < 25:
        return None
    k = np.where((US > 0.8) & (vals < tail + level * (pl - tail)))[0]
    if not len(k) or US[k[0]] >= US[ok][-1] - 0.05:
        return None
    return float(US[k[0]])


def edge_peak(vals, ok, level=0.5):
    if ok.sum() < 60:
        return None
    seg = (US >= 0.5) & (US <= 1.6) & ok
    if not seg.any():
        return None
    j = int(np.where(seg)[0][np.argmax(vals[seg])]); pk, tail = float(vals[j]), tail_of(vals, ok)
    if pk - tail < 25:
        return None
    k = np.where((US > US[j]) & (vals < tail + level * (pk - tail)))[0]
    if not len(k) or US[k[0]] >= US[ok][-1] - 0.05:
        return None
    return float(US[k[0]])


def measure_track(mat, ts, t, anns):
    T, N = mat.shape
    nyq, bf, rbf, sem = t['nyquist_mps'], t['baseline_frac'], t['spectral_row_baseline_frac'], t['semantic_id']
    cw = sem == 'continuous_wave'
    zero, per = rbf * (N - 1), 2.0 * nyq / (N - 1)
    bg = np.percentile(mat, 20, axis=0)
    out = []
    for a in anns:
        kind = lab_of(a.get('label'))
        pts = a.get('points') or []
        if kind == 'MV':
            if len(pts) != 3 or abs(pts[2]['velocity_mps']) < 0.05:
                continue
            use = [('E', pts[0]), ('A', pts[2])]
        elif kind in ('e_sep', 'e_lat'):
            use = [(kind, pts[0])] if len(pts) == 1 else []
        elif kind == 'TR':
            use = [('TR', pts[0])] if (len(pts) == 1 and cw) else []
        else:
            continue
        for name, p in use:
            v = corrected(p['velocity_mps'], nyq, bf, rbf, sem)
            d = abs(v) / per
            rec = {'name': name, 'v': abs(float(v)), 'N': int(N), 'nyq': float(nyq), 'mism': int(abs(bf - rbf) > 1e-9),
                   'r50': None, 'r25': None, 'r75': None, 'rpk': None}
            if d >= 5:
                col = int(np.argmin(np.abs(ts - p['time_s']))); w = 6 if cw else 1
                prof = np.convolve(mat[max(0, col - w):col + w + 1].max(axis=0) - bg, np.ones(3) / 3, mode='same')
                vals, ok = sample(prof, zero, 1 if v > 0 else -1, d)
                rec['r50'], rec['r25'], rec['r75'] = (edge_plateau(vals, ok, L) for L in (0.5, 0.25, 0.75))
                rec['rpk'] = edge_peak(vals, ok)
            out.append(rec)
    return out


def cohort():
    X = RR.load_external()
    ee = sorted(e for e, r in X.items() if not r['af'] and r['pred'] and r['y_ee'] is not None)
    main = {e for e in ee if X[e]['y_lap'] is not None and X[e]['n_avail'] == 3}
    return X, ee, main


def measure():
    import zarr
    X, ee, main = cohort()
    print(f'E/e′ kümesi {len(ee)}, ana küme {len(main)}', flush=True)
    rows = []
    for n, ex in enumerate(ee, 1):
        path = os.path.join(ROOT, 'echoxflow', 'exams', ex + '.tar')
        want = {}
        with tarfile.open(path) as tf:                                  # 1. geçiş: hangi kayıtlar gerekli
            for m in tf:
                if m.name.endswith('.zarr/.zattrs'):
                    man = json.load(tf.extractfile(m)).get('recording_manifest', {})
                    if any(lab_of(a.get('label')) for t in (man.get('tracks') or []) for a in (t.get('spectral_annotations') or [])):
                        want[m.name[:-len('.zattrs')]] = man
        dest = os.path.join(TMP, ex); shutil.rmtree(dest, ignore_errors=True)
        if want:
            with tarfile.open(path) as tf:                              # 2. geçiş: yalnız spektrum ve zaman dizileri
                keep = [m for m in tf if any(m.name.startswith(w) for w in want) and
                        (m.name.rsplit('/', 1)[-1].startswith('.z') and m.name.count('/') <= 4
                         or '/data/1d_pulsed_wave_doppler/' in m.name or '/data/1d_continuous_wave_doppler/' in m.name
                         or '/timestamps/1d_pulsed_wave_doppler/' in m.name or '/timestamps/1d_continuous_wave_doppler/' in m.name
                         or m.isdir())]
                tf.extractall(dest, members=keep, filter='data')
        for w, man in want.items():
            zp = os.path.join(dest, w.rstrip('/'))
            try:
                z = zarr.open(zp, mode='r')
            except Exception:
                continue
            for t in man.get('tracks') or []:
                anns = [a for a in (t.get('spectral_annotations') or []) if lab_of(a.get('label'))]
                if not anns or 'nyquist_mps' not in t or t.get('baseline_frac') is None:
                    continue
                try:
                    mat = np.asarray(z[t['data']['zarr_path']]).astype(np.float32)
                    ts = np.asarray(z[t['timestamps']['zarr_path']]).ravel()
                except Exception:
                    continue
                for r in measure_track(mat, ts, t, anns):
                    r.update(exam_id=ex, rid=os.path.basename(zp)[:-5], main=int(ex in main)); rows.append(r)
        shutil.rmtree(dest, ignore_errors=True)
        if n % 25 == 0 or n == len(ee):
            print(f'[{n}/{len(ee)}] kaliper {len(rows)}', flush=True)
    cols = ['exam_id', 'rid', 'main', 'name', 'v', 'N', 'nyq', 'mism', 'r50', 'r25', 'r75', 'rpk']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    summarize()


# ------------------------------------------------------------------ özet
def boot_med(xs, rng):
    xs = np.asarray(xs, float)
    if len(xs) < 5:
        return [None, None]
    b = np.median(xs[rng.integers(0, len(xs), (N_BOOT, len(xs)))], axis=1)
    return [R4(np.percentile(b, 2.5)), R4(np.percentile(b, 97.5))]


def exam_medians(rows, names, key='r50', sel=None):
    by = {}
    for r in rows:
        if r['name'] in names and r[key] is not None and (sel is None or sel(r)):
            by.setdefault(r['exam_id'], []).append(r[key])
    return {e: float(np.median(v)) for e, v in by.items()}


def block(rows, names, key='r50', sel=None):
    em = exam_medians(rows, names, key, sel); xs = list(em.values())
    tried = sum(1 for r in rows if r['name'] in names and (sel is None or sel(r)))
    valid = sum(1 for r in rows if r['name'] in names and r[key] is not None and (sel is None or sel(r)))
    if not xs:
        return {'kaliper': tried, 'gecerli_kaliper': valid, 'muayene': 0}
    return {'kaliper': tried, 'gecerli_kaliper': valid, 'muayene': len(xs), 'oran_medyan': R4(np.median(xs)),
            'iqr': [R4(np.percentile(xs, 25)), R4(np.percentile(xs, 75))],
            'ga': boot_med(xs, np.random.default_rng(SEED)),
            'orani_1den_buyuk_muayene': RR.prop(int(sum(x > 1.0 for x in xs)), len(xs))}


def theil(rows, names, unit):
    from scipy.stats import theilslopes
    pts = [(r['exam_id'], r['v'] * unit, r['r50'] * r['v'] * unit) for r in rows if r['name'] in names and r['r50'] is not None]
    if len(pts) < 30:
        return None
    x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
    b, a, _, _ = theilslopes(y, x)
    ex = sorted({p[0] for p in pts}); idx = {e: [i for i, p in enumerate(pts) if p[0] == e] for e in ex}
    rng = np.random.default_rng(SEED); bs, as_ = [], []
    for _ in range(500):                                               # muayene kümeli bootstrap; hız için ≤700 kaliper
        ii = np.concatenate([idx[ex[j]] for j in rng.integers(0, len(ex), len(ex))])
        if len(ii) > 700:
            ii = rng.choice(ii, 700, replace=False)
        bb, aa, _, _ = theilslopes(y[ii], x[ii]); bs.append(bb); as_.append(aa)
    return {'n_kaliper': len(pts), 'n_muayene': len(ex), 'isaret_p5_p95': [R4(np.percentile(x, 5)), R4(np.percentile(x, 95))],
            'egim_b': R4(b), 'egim_ga': [R4(np.percentile(bs, 2.5)), R4(np.percentile(bs, 97.5))],
            'sabit_a': R4(a), 'sabit_ga': [R4(np.percentile(as_, 2.5)), R4(np.percentile(as_, 97.5))],
            'birim': 'cm/s' if unit == 100 else 'm/s'}


def summarize():
    f = lambda s: None if s in ('', 'None') else float(s)
    rows = []
    for r in csv.DictReader(open(OUT_CSV)):
        rows.append({'exam_id': r['exam_id'], 'main': int(r['main']), 'name': r['name'], 'v': float(r['v']), 'N': int(r['N']),
                     'nyq': float(r['nyq']), 'mism': int(r['mism']), **{k: f(r[k]) for k in ('r50', 'r25', 'r75', 'rpk')}})
    FLOW, TDI = ('E', 'A'), ('e_sep', 'e_lat')
    out = {'aciklama': 'oran = zarf kenarı / işaret (taban çizgisinden uzaklık); birincil kestirici plato-referanslı yarı-düşüş (r50)',
           'kohort_muayene': len({r['exam_id'] for r in rows}), 'kaliper_toplam': len(rows),
           'birincil': {'akim_E_A': block(rows, FLOW), 'doku_e': block(rows, TDI), 'TR_kontrol': block(rows, ('TR',)),
                        'E': block(rows, ('E',)), 'A': block(rows, ('A',)),
                        'e_septal': block(rows, ('e_sep',)), 'e_lateral': block(rows, ('e_lat',))},
           'ana_kume': {'akim_E_A': block(rows, FLOW, sel=lambda r: r['main']), 'doku_e': block(rows, TDI, sel=lambda r: r['main']),
                        'TR_kontrol': block(rows, ('TR',), sel=lambda r: r['main'])},
           'kestirici_duyarliligi': {k: {'akim_E_A': block(rows, FLOW, key=k), 'doku_e': block(rows, TDI, key=k),
                                         'TR_kontrol': block(rows, ('TR',), key=k)} for k in ('r25', 'r75', 'rpk')},
           'alt_gruplar': {
               'akim_satir_128': block(rows, FLOW, sel=lambda r: r['N'] == 128), 'akim_satir_diger': block(rows, FLOW, sel=lambda r: r['N'] != 128),
               'doku_satir_96': block(rows, TDI, sel=lambda r: r['N'] == 96), 'doku_satir_diger': block(rows, TDI, sel=lambda r: r['N'] != 96),
               'akim_bf_esit_rbf': block(rows, FLOW, sel=lambda r: not r['mism']), 'akim_bf_farkli_rbf': block(rows, FLOW, sel=lambda r: r['mism']),
               'doku_bf_esit_rbf': block(rows, TDI, sel=lambda r: not r['mism']), 'doku_bf_farkli_rbf': block(rows, TDI, sel=lambda r: r['mism']),
               'akim_nyq_dusuk': block(rows, FLOW, sel=lambda r: r['nyq'] < 0.7), 'akim_nyq_yuksek': block(rows, FLOW, sel=lambda r: r['nyq'] >= 0.7),
               'doku_nyq_015': block(rows, TDI, sel=lambda r: abs(r['nyq'] - 0.15) < 1e-6), 'doku_nyq_diger': block(rows, TDI, sel=lambda r: abs(r['nyq'] - 0.15) >= 1e-6)},
           'dogrusal_uyum': {'akim_E_A': theil(rows, FLOW, 1), 'doku_e': theil(rows, TDI, 100), 'TR_kontrol': theil(rows, ('TR',), 1)}}
    # hız üçte birleri (çarpımsal mı, toplamsal mı)
    for nm, names, unit in (('akim_E_A', FLOW, 1), ('doku_e', TDI, 100)):
        pts = np.array([(r['v'] * unit, r['r50']) for r in rows if r['name'] in names and r['r50'] is not None])
        if len(pts) >= 30:
            q1, q2 = np.percentile(pts[:, 0], [33.3, 66.7]); lo, hi = pts[pts[:, 0] <= q1], pts[pts[:, 0] >= q2]
            out['dogrusal_uyum'][nm]['hiz_uctebir'] = {
                'dusuk': {'n': len(lo), 'isaret_medyan': R4(np.median(lo[:, 0])), 'oran_medyan': R4(np.median(lo[:, 1])),
                          'fark_medyan': R4(np.median(lo[:, 0] * (lo[:, 1] - 1)))},
                'yuksek': {'n': len(hi), 'isaret_medyan': R4(np.median(hi[:, 0])), 'oran_medyan': R4(np.median(hi[:, 1])),
                           'fark_medyan': R4(np.median(hi[:, 0] * (hi[:, 1] - 1)))}}
    # aynı muayenede akım / doku
    a, b = exam_medians(rows, FLOW), exam_medians(rows, TDI)
    both = sorted(set(a) & set(b)); q = [a[e] / b[e] for e in both]
    out['akim_bolu_doku_ayni_muayene'] = {'muayene': len(both), 'oran_medyan': R4(np.median(q)) if q else None,
                                          'iqr': [R4(np.percentile(q, 25)), R4(np.percentile(q, 75))] if q else None,
                                          'ga': boot_med(q, np.random.default_rng(SEED))}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res.update({'uretim': 'external/posthoc/pw_scale_check.py', 'tohum': SEED, 'bootstrap_tekrar': N_BOOT,
                'not': 'POST HOC (analysis_log Ek 26). Kestiricinin yanlılığı düzeltilmedi; TR (CW) kontroldür.', 'olcum': out})
    json.dump(res, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    p = out['birincil']
    for k in ('akim_E_A', 'doku_e', 'TR_kontrol', 'E', 'A', 'e_septal', 'e_lateral'):
        d = p[k]; print(f"{k:12s} kaliper {d['gecerli_kaliper']}/{d['kaliper']} muayene {d['muayene']} oran {d.get('oran_medyan')} IQR {d.get('iqr')} GA {d.get('ga')}")
    print('akım/doku (aynı muayene):', out['akim_bolu_doku_ayni_muayene'])
    print('doğrusal uyum:', json.dumps(out['dogrusal_uyum'], ensure_ascii=False)[:900])


# ------------------------------------------------------------------ duyarlılık senaryoları
def scenarios():
    res = json.load(open(OUT_JSON)); M = res['olcum']['birincil']
    Ff, Ft = M['akim_E_A']['oran_medyan'], M['doku_e']['oran_medyan']
    B = json.load(open(RR.H + 'runs/b2_binary/binary_results.json'))
    L = {'thr': B['esik'], 'lo': B['gri_bolge']['rule_out_alti'], 'hi': B['gri_bolge']['rule_in_ustu']}
    I = RR.load_internal(fix=False); va = RR.pick(I, 'val')
    L['thr95'] = RR.val_threshold(np.array([r['y'] for r in va]), np.array([r['p'] for r in va]), 0.95)
    X, ee, main = cohort()
    ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(RR.EXT, 'echoxflow_reference_v2.csv')))}
    f = lambda s: None if s in ('', 'None') else float(s)

    def labels(fE, fe):
        out = {}
        for e in ee:
            r = ref[e]; E, A, es, el = f(r['E']), f(r['A']), f(r['e_sep']), f(r['e_lat'])
            p = {'E': E * fE if E is not None else None, 'A': A * fE if A is not None else None,
                 'e_sep': es * fe if es is not None else None, 'e_lat': el * fe if el is not None else None, 'TRvel': f(r['TRvel'])}
            p['EA'] = (p['E'] / p['A']) if (p['E'] and p['A'] and p['A'] > 1.0) else None
            p = derive(p); g = grade_ase2025(p, age=None)
            s_, l_, m_ = ee_value(p, 'sep'), ee_value(p, 'lat'), ee_value(p, 'mean')
            y_ee = int((s_ is not None and s_ >= 15) or (l_ is not None and l_ >= 13) or (m_ is not None and m_ >= 14))
            y_lap = None if g.get('lap') is None else int(g['lap'] == 'elevated')
            out[e] = (y_ee, y_lap, bool((g.get('primary') or {}).get('e_reduced')))
        return out
    S0 = labels(1.0, 1.0)
    assert all(S0[e][0] == X[e]['y_ee'] and S0[e][1] == X[e]['y_lap'] for e in ee), 'S0 mevcut referansı yeniden üretmiyor'
    sc = {}
    for name, fE, fe, acik in (('S0_disa_aktarilan', 1.0, 1.0, 'dışa aktarılan değerler (mevcut referans)'),
                               ('S1_ortak_carpan', Ff, Ff, f'E, A ve e′ × {Ff}'),
                               ('S2_ayri_carpan', Ff, Ft, f'E, A × {Ff}; e′ × {Ft}')):
        lab = labels(fE, fe); d = {'aciklama': acik, 'carpan_akim': fE, 'carpan_doku': fe}
        y = np.array([lab[e][0] for e in ee]); p = np.array([X[e]['p'] for e in ee])
        d['Ee'] = {**RR.describe(y, p, L), 'S0a_gore_degisen_etiket': int(sum(lab[e][0] != S0[e][0] for e in ee))}
        ms = [e for e in ee if lab[e][1] is not None and X[e]['n_avail'] == 3]
        y = np.array([lab[e][1] for e in ms]); p = np.array([X[e]['p'] for e in ms])
        d['kilavuz_uc_olcum'] = {**RR.describe(y, p, L),
                                 'S0_ana_kumesinden_kalan': int(sum(e in main for e in ms)), 'yeni_giren': int(sum(e not in main for e in ms)),
                                 'S0_olaylarindan_kalan': int(sum(S0[e][1] == 1 and e in main and lab[e][1] == 1 for e in ms)),
                                 'kalanlarda_degisen_etiket': int(sum(e in main and lab[e][1] != S0[e][1] for e in ms))}
        d['e_azalmis_muayene'] = int(sum(lab[e][2] for e in ee))
        sc[name] = d
        print(f"{name:20s} E/e′ n={d['Ee']['n']} olay={d['Ee']['olay']} AUROC={d['Ee']['auroc']} {d['Ee']['ga']} | "
              f"kılavuz n={d['kilavuz_uc_olcum']['n']} olay={d['kilavuz_uc_olcum']['olay']} AUROC={d['kilavuz_uc_olcum']['auroc']} "
              f"{d['kilavuz_uc_olcum']['ga']} dışlanan={d['kilavuz_uc_olcum']['uc_bolge']['dislanan']['oran']} "
              f"kaçan={d['kilavuz_uc_olcum']['uc_bolge']['dislananlarda_kacirilan_olay']}", flush=True)
    res['senaryolar'] = sc
    json.dump(res, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    {'measure': measure, 'summarize': summarize, 'scenarios': scenarios}[sys.argv[1]]()
