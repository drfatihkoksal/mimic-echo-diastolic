#!/usr/bin/env python3
"""Doku Doppler e′: kayıtlı değer ile görüntüden ölçülen zarfın karşılaştırması, iki veri setinde aynı kaliper-bağımsız
kestiriciyle (analysis_log Ek 27; post hoc). MIMIC, tarayıcının kendi DICOM kalibrasyonuyla negatif kontroldür.

  python3 external/posthoc/tdi_scale_check.py mimic      -> ~/mimic-echo/rev_scale/mimic_tdi.csv   (proje dışı, DUA)
  python3 external/posthoc/tdi_scale_check.py echox      -> external/posthoc/tdi_scale_echox.csv
  python3 external/posthoc/tdi_scale_check.py summarize  -> external/posthoc/tdi_scale_results.json (yalnız toplu sayılar)

MIMIC görüntüleri hiçbir çıktıya yazılmaz. Çalıştırma: proje kökünden."""
from __future__ import annotations
import csv, glob, json, os, shutil, sys, tarfile, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(ROOT, 'echoxflow-code', 'src'))
H = os.path.expanduser('~/mimic-echo/')
DICOM_ROOT = os.environ.get('MIMIC_ECHO_DICOM_ROOT', os.path.expanduser('~/mimic-echo/dicom'))
OUT_M = os.path.join(H, 'rev_scale', 'mimic_tdi.csv')
OUT_X = os.path.join(HERE, 'tdi_scale_echox.csv')
OUT_J = os.path.join(HERE, 'tdi_scale_results.json')
SEED, N_BOOT = 20260901, 4000
DV, DT = 0.25, 0.004                      # ortak ızgara: cm/s, s
LEVELS = (0.5, 0.25, 0.75)
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)


# ------------------------------------------------------------------ ortak kestirici
def to_grid(S, v, t):
    """S[satır(hız), sütun(zaman)], v (cm/s, satır başına), t (s, sütun başına) -> ortak ızgara; satır 0 = en pozitif hız."""
    import cv2
    v = np.asarray(v, float); t = np.asarray(t, float)
    if v[0] < v[-1]:
        S, v = S[::-1], v[::-1]
    vmax, vmin = float(v[0]), float(v[-1])
    nv = int(round((vmax - vmin) / DV)) + 1; nt = int(round((t[-1] - t[0]) / DT)) + 1
    if nv < 20 or nt < 100:
        return None, None
    G = cv2.resize(S.astype(np.float32), (nt, nv), interpolation=cv2.INTER_AREA if (nv < S.shape[0] or nt < S.shape[1]) else cv2.INTER_LINEAR)
    return G, vmax - DV * np.arange(nv)


def envelopes(G, vg):
    """Döner: {L: (pozitif zarf, negatif zarf)} cm/s, sütun başına (zarf yoksa 0)."""
    import cv2
    G = G - np.percentile(G, 20, axis=1, keepdims=True)
    G = cv2.medianBlur(np.clip(G, 0, 255).astype(np.uint8), 3).astype(np.float32) if G.shape[1] > 3 else G
    k = max(1, int(round(1.0 / DV)))
    G = cv2.blur(G, (1, k))                                                  # hızda 1 cm/s ortalama
    out = {L: (np.zeros(G.shape[1]), np.zeros(G.shape[1])) for L in LEVELS}
    for side in (0, 1):
        idx = np.where(vg >= 2.0)[0][::-1] if side == 0 else np.where(vg <= -2.0)[0]   # taban çizgisinden dışa doğru
        if len(idx) < 12:
            continue
        P = G[idx]                                                           # (uzaklık, zaman)
        nt_ = max(3, int(0.12 * len(idx)))
        tail = np.median(P[-nt_:], axis=0); j = P.argmax(axis=0); pk = P[j, np.arange(P.shape[1])]
        dist = np.abs(vg[idx])
        ok = (pk - tail) >= 25
        for L in LEVELS:
            thr = tail + L * (pk - tail)
            below = (P < thr[None, :]) & (np.arange(len(idx))[:, None] > j[None, :])
            first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
            env = np.where(ok & (first > 0) & (first < len(idx) - 1), dist[np.clip(first, 0, len(idx) - 1)], 0.0)
            out[L][side][:] = env
    return out


def waves(pos, neg):
    """s′ tepeleri arasından döngüler; e′ = aralığın 0,25–0,60 bölümündeki en büyük negatif zarf, a′ = 0,62–0,97."""
    from scipy.signal import find_peaks
    k = max(1, int(round(0.03 / DT))); sm = np.convolve(pos, np.ones(k) / k, mode='same')
    if sm.max() <= 0:
        return None
    pk, _ = find_peaks(sm, distance=int(0.4 / DT), height=0.5 * np.percentile(sm, 95))
    es, as_, ss, cyc = [], [], [], []
    for a, b in zip(pk, pk[1:]):
        n = b - a
        if not (0.4 <= n * DT <= 1.6):
            continue
        e = neg[a + int(0.25 * n): a + int(0.60 * n)]; aw = neg[a + int(0.62 * n): a + int(0.97 * n)]
        if len(e) and e.max() > 0:
            es.append(float(e.max())); as_.append(float(aw.max()) if len(aw) else 0.0); ss.append(float(sm[a])); cyc.append(n * DT)
    if not es:
        return None
    return {'e': float(np.median(es)), 'a': float(np.median(as_)), 's': float(np.median(ss)), 'cycle': float(np.median(cyc)), 'n': len(es)}


def analyse(S, v, t):
    G, vg = to_grid(S, v, t)
    if G is None:
        return None
    env = envelopes(G, vg); res = {}
    for L in LEVELS:
        w = waves(env[0.5][0], env[L][1])                                    # döngüler birincil düzeydeki s′ ile
        if w is None:
            return None
        res[L] = w
    return res


# ------------------------------------------------------------------ MIMIC
def mimic_job(a):
    sid, path = a
    import pydicom, cv2
    try:
        hdr = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return []
    regs = [r for r in (getattr(hdr, 'SequenceOfUltrasoundRegions', None) or []) if int(getattr(r, 'RegionDataType', -1)) == 3]
    if not regs or int(getattr(hdr, 'NumberOfFrames', 1) or 1) != 1:
        return []
    r = regs[0]
    try:
        x0, x1, y0, y1 = int(r.RegionLocationMinX0), int(r.RegionLocationMaxX1), int(r.RegionLocationMinY0), int(r.RegionLocationMaxY1)
        dy, dx, ry = float(r.PhysicalDeltaY), float(r.PhysicalDeltaX), float(getattr(r, 'ReferencePixelY0', 0))
        if int(r.PhysicalUnitsYDirection) != 7 or int(r.PhysicalUnitsXDirection) != 4 or y1 - y0 < 60 or x1 - x0 < 200 or dy == 0 or dx <= 0:
            return []
    except Exception:
        return []
    # Ekran koordinatında satır aşağı doğru artar, görüntülenen hız yukarı doğru pozitiftir: v = (taban satırı − satır)·|ΔY|.
    # (İlk koşuda (satır − taban)·ΔY kullanıldı; ΔY bütün görüntülerde pozitif olduğu için eksen ters çevrilmişti. Ek 27 düzeltmesi.)
    v = (ry - np.arange(y1 - y0)) * abs(dy)                                   # cm/s
    if max(abs(v[0]), abs(v[-1])) > 40:                                       # doku Doppler: görüntülenen aralık ≤40 cm/s
        return []
    try:
        a_ = pydicom.dcmread(path).pixel_array
    except Exception:
        return []
    if a_.ndim == 4:
        a_ = a_[0]
    roi = a_[y0:y1, x0:x1]
    if roi.ndim == 3:
        hsv = cv2.cvtColor(np.ascontiguousarray(roi[..., :3]).astype(np.uint8), cv2.COLOR_RGB2HSV)
        gray = (0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]).astype(np.float32)
        br = (hsv[..., 2] > 100) & (hsv[..., 1] > 40)
        if br.sum() > 200:                                                    # bindirme grafikleri: baskın tondan farklı doygun pikseller
            hh = np.bincount(hsv[..., 0][br], minlength=180); dom = int(hh.argmax())
            dh = np.abs(hsv[..., 0].astype(int) - dom); dh = np.minimum(dh, 180 - dh)
            gray[(hsv[..., 1] > 90) & (dh > 20)] = 0
        else:
            gray[hsv[..., 1] > 90] = 0
    else:
        gray = roi.astype(np.float32)
    res = analyse(gray, v, np.arange(x1 - x0) * dx)
    if res is None:
        return [{'study_id': sid, 'ok': 0}]
    row = {'study_id': sid, 'ok': 1, 'vmax': round(float(max(abs(v[0]), abs(v[-1]))), 2)}
    for L in LEVELS:
        row.update({f'e{int(L * 100)}': round(res[L]['e'], 3), f'a{int(L * 100)}': round(res[L]['a'], 3)})
    row.update(s=round(res[0.5]['s'], 3), cycle=round(res[0.5]['cycle'], 3), n=res[0.5]['n'])
    return [row]


def mimic():
    os.makedirs(os.path.dirname(OUT_M), exist_ok=True)
    sp = {r['study_id']: r['split'] for r in csv.DictReader(open(H + 'splits_binary.csv'))}
    lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
    test = sorted(s for s, v in sp.items() if v == 'test' and lab.get(s, {}).get('e_sep') not in ('', None) and lab[s].get('e_lat') not in ('', None))
    want = set(test); jobs = []
    for r in csv.DictReader(open(H + 'echo-record-list.csv')):
        if r['study_id'] in want:
            p = os.path.join(DICOM_ROOT, r['dicom_filepath'])
            if not os.path.exists(p):
                p = os.path.join(os.path.dirname(DICOM_ROOT), r['dicom_filepath'])
            jobs.append((r['study_id'], p))
    print(f'test çalışması (septal ve lateral e′ kayıtlı): {len(test)}; DICOM: {len(jobs)}', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, out in enumerate(ex.map(mimic_job, jobs, chunksize=64), 1):
            rows += out
            if i % 5000 == 0:
                print(f'  {i}/{len(jobs)} doku Doppler görüntüsü {len(rows)}', flush=True)
    cols = ['study_id', 'ok', 'vmax', 'e50', 'a50', 'e25', 'a25', 'e75', 'a75', 's', 'cycle', 'n']
    with open(OUT_M, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f'doku Doppler görüntüsü {len(rows)}, ölçülebilen {sum(r["ok"] for r in rows)} -> {OUT_M}', flush=True)


# ------------------------------------------------------------------ EchoXFlow
def echox():
    import zarr
    import pw_scale_check as PW
    X, ee, main = PW.cohort()
    TMP = os.path.expanduser('~/echoxflow-render/_tdi'); rows = []
    for n, ex in enumerate(ee, 1):
        path = os.path.join(ROOT, 'echoxflow', 'exams', ex + '.tar'); want = {}
        with tarfile.open(path) as tf:
            for m in tf:
                if m.name.endswith('.zarr/.zattrs'):
                    man = json.load(tf.extractfile(m)).get('recording_manifest', {})
                    if any('MV Eprime Septal' in (a.get('label') or '') or 'MV Eprime Lateral' in (a.get('label') or '')
                           for t in (man.get('tracks') or []) for a in (t.get('spectral_annotations') or [])):
                        want[m.name[:-len('.zattrs')]] = man
        dest = os.path.join(TMP, ex); shutil.rmtree(dest, ignore_errors=True)
        if want:
            with tarfile.open(path) as tf:
                keep = [m for m in tf if any(m.name.startswith(w) for w in want) and
                        (m.name.rsplit('/', 1)[-1].startswith('.z') and m.name.count('/') <= 4 or m.isdir()
                         or '/data/1d_pulsed_wave_doppler/' in m.name or '/timestamps/1d_pulsed_wave_doppler/' in m.name)]
                tf.extractall(dest, members=keep, filter='data')
        for w, man in want.items():
            zp = os.path.join(dest, w.rstrip('/'))
            for t in man.get('tracks') or []:
                anns = [a for a in (t.get('spectral_annotations') or []) if 'MV Eprime' in (a.get('label') or '')]
                if not anns or t.get('semantic_id') != 'pw_doppler' or 'nyquist_mps' not in t:
                    continue
                try:
                    z = zarr.open(zp, mode='r')
                    mat = np.asarray(z[t['data']['zarr_path']]).astype(np.float32); ts = np.asarray(z[t['timestamps']['zarr_path']]).ravel()
                except Exception:
                    continue
                N = mat.shape[1]; v = (t['spectral_row_baseline_frac'] * (N - 1) - np.arange(N)) * 2.0 * t['nyquist_mps'] / (N - 1) * 100.0
                site = 'sep' if 'Septal' in anns[0]['label'] else 'lat'
                cal = float(np.median([abs(p['velocity_mps']) * 100 for a in anns for p in a['points']]))
                res = analyse(mat.T, v, ts - ts[0])
                row = {'exam_id': ex, 'site': site, 'caliper': round(cal, 3), 'ok': int(res is not None), 'vmax': round(float(max(abs(v[0]), abs(v[-1]))), 2)}
                if res:
                    for L in LEVELS:
                        row.update({f'e{int(L * 100)}': round(res[L]['e'], 3), f'a{int(L * 100)}': round(res[L]['a'], 3)})
                    row.update(s=round(res[0.5]['s'], 3), cycle=round(res[0.5]['cycle'], 3), n=res[0.5]['n'])
                rows.append(row)
        shutil.rmtree(dest, ignore_errors=True)
        if n % 50 == 0 or n == len(ee):
            print(f'[{n}/{len(ee)}] kayıt {len(rows)}', flush=True)
    cols = ['exam_id', 'site', 'caliper', 'ok', 'vmax', 'e50', 'a50', 'e25', 'a25', 'e75', 'a75', 's', 'cycle', 'n']
    with open(OUT_X, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f'kayıt {len(rows)}, ölçülebilen {sum(r["ok"] for r in rows)} -> {OUT_X}', flush=True)


# ------------------------------------------------------------------ özet
def boot(fn, n, rng):
    b = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [R4(np.percentile(b, 2.5)), R4(np.percentile(b, 97.5))]


def summarize():
    from scipy.stats import spearmanr
    f = lambda s: None if s in ('', 'None', None) else float(s)
    lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
    M = {}
    for r in csv.DictReader(open(OUT_M)):
        if r['ok'] == '1':
            M.setdefault(r['study_id'], []).append(r)
    Xr = {}
    for r in csv.DictReader(open(OUT_X)):
        Xr.setdefault(r['exam_id'], []).append(r)
    ximg_all = sum(len(v) for v in Xr.values())
    mimg_all = sum(1 for _ in csv.DictReader(open(OUT_M)))
    res = {'uretim': 'external/posthoc/tdi_scale_check.py', 'tohum': SEED, 'bootstrap_tekrar': N_BOOT,
           'not': 'POST HOC (analysis_log Ek 27). oran = görüntüden ölçülen e′ / kayıtlı septal ve lateral e′ ortalaması.', 'duzeyler': {}}
    for L in LEVELS:
        k = f'e{int(L * 100)}'
        m = [(float(np.median([float(r[k]) for r in rs])), (f(lab[s]['e_sep']) + f(lab[s]['e_lat'])) / 2.0, len(rs)) for s, rs in M.items()]
        m = np.array([(a, b) for a, b, _ in m if b and b > 0])
        x = []
        for e, rs in Xr.items():
            ok = [r for r in rs if r['ok'] == '1']
            sites = {r['site'] for r in rs}
            if not ok or sites != {'sep', 'lat'}:
                continue
            cal = np.mean([np.median([float(r['caliper']) for r in rs if r['site'] == s_]) for s_ in ('sep', 'lat')])
            x.append((float(np.median([float(r[k]) for r in ok])), cal))
        x = np.array(x)
        rm, rx = m[:, 0] / m[:, 1], x[:, 0] / x[:, 1]
        rng = np.random.default_rng(SEED)
        d = {'mimic': {'calisma': int(len(m)), 'oran_medyan': R4(np.median(rm)), 'iqr': [R4(np.percentile(rm, 25)), R4(np.percentile(rm, 75))],
                       'ga': boot(lambda i: np.median(rm[i]), len(rm), rng), 'spearman': R4(spearmanr(m[:, 0], m[:, 1])[0]),
                       'goruntu_e_medyan': R4(np.median(m[:, 0])), 'kayitli_e_medyan': R4(np.median(m[:, 1]))},
             'echoxflow': {'muayene': int(len(x)), 'oran_medyan': R4(np.median(rx)), 'iqr': [R4(np.percentile(rx, 25)), R4(np.percentile(rx, 75))],
                           'ga': boot(lambda i: np.median(rx[i]), len(rx), rng), 'spearman': R4(spearmanr(x[:, 0], x[:, 1])[0]),
                           'goruntu_e_medyan': R4(np.median(x[:, 0])), 'kayitli_e_medyan': R4(np.median(x[:, 1]))}}
        rng = np.random.default_rng(SEED)
        bq = [np.median(rx[rng.integers(0, len(rx), len(rx))]) / np.median(rm[rng.integers(0, len(rm), len(rm))]) for _ in range(N_BOOT)]
        d['echoxflow_bolu_mimic'] = {'bolum': R4(np.median(rx) / np.median(rm)), 'ga': [R4(np.percentile(bq, 2.5)), R4(np.percentile(bq, 97.5))]}
        res['duzeyler'][f'L{int(L * 100)}'] = d
        print(f"L={L}: MIMIC n={len(m)} oran {d['mimic']['oran_medyan']} {d['mimic']['ga']} ρ={d['mimic']['spearman']} | "
              f"EchoXFlow n={len(x)} oran {d['echoxflow']['oran_medyan']} {d['echoxflow']['ga']} ρ={d['echoxflow']['spearman']} | "
              f"bölüm {d['echoxflow_bolu_mimic']['bolum']} {d['echoxflow_bolu_mimic']['ga']}")
    res['islem'] = {'mimic_doku_goruntusu': mimg_all, 'mimic_olculebilen_goruntu': int(sum(len(v) for v in M.values())),
                    'echoxflow_kayit': ximg_all, 'echoxflow_olculebilen_kayit': int(sum(1 for v in Xr.values() for r in v if r['ok'] == '1'))}
    # ikincil tanımlayıcılar (yöntemin akla yatkınlığı): s′, döngü süresi
    for nm, rows in (('mimic', [r for v in M.values() for r in v]), ('echoxflow', [r for v in Xr.values() for r in v if r['ok'] == '1'])):
        res['islem'][nm + '_s_medyan'] = R4(np.median([float(r['s']) for r in rows])); res['islem'][nm + '_dongu_s_medyan'] = R4(np.median([float(r['cycle']) for r in rows]))
        res['islem'][nm + '_gorunen_aralik_medyan'] = R4(np.median([float(r['vmax']) for r in rows]))
    json.dump(res, open(OUT_J, 'w'), ensure_ascii=False, indent=1)
    print(res['islem'])


if __name__ == '__main__':
    {'mimic': mimic, 'echox': echox, 'summarize': summarize}[sys.argv[1]]()
