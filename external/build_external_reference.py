#!/usr/bin/env python3
"""
Dış validasyon — Adım 1: EchoXFlow (Ahus, Norveç) için referans standardın kurulması.

EchoXFlow muayenelerinde klinisyenin yaptığı Doppler ölçümleri, her kaydın Zarr
manifest'inde `tracks[].spectral_annotations` altında fiziksel birimlerle duruyor.
Bu betik 411 GB'ı açmadan, yalnızca tar başlıklarındaki .zattrs'leri okuyarak
muayene düzeyinde bir ölçüm tablosu ve 2025 ASE ikili son noktasını (yüksek
dolum basıncı var/yok) üretir.

DÜZELTME (önemli; YALNIZ CW — PW'ye uygulanmaz, bkz. corrected()): CW izlerinde `baseline_frac` ile `spectral_row_baseline_frac`
tutarsız (0.1 vs 0.5); üreticinin dönüştürücüsü 0.5 varsaydığı için hızlar
sabit bir ofset kadar kaymış. Düzeltme:  v_true = v_rapor + (rbf - bf) * 2 * nyquist
Doğrulama: düzeltmesiz TR Vmax medyanı 5.87 m/s (fizyolojik değil),
düzeltmeyle 2.70 m/s (IQR 2.37-3.05) — beklenen dağılım.
"""
from __future__ import annotations
import csv, glob, json, os, statistics as st, sys, tarfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grading'))
from diastolic_grading import derive, grade_ase2025, ee_value  # noqa: E402

EXAM_TARS = 'echoxflow/exams/*.tar'

# Etiketten kanonik ölçüme eşleme. (nokta sayısı, hangi noktalar)
SPECTRAL = {
    'Cardiac/SD/Mitral Valve/MV E/A Velocity':        ('EA_pair', 3),
    'Cardiac/SD/Mitral Valve/MV Eprime Septal Velocity':  ('e_sep', 1),
    'Cardiac/SD/Mitral Valve/MV Eprime Lateral Velocity': ('e_lat', 1),
    'Cardiac/SD/Mitral Valve/MV Eprime Velocity':         ('e_unspec', 1),
    'Cardiac/SD/Tricuspid Valve/TR Vmax':                 ('TRvel', 1),
    'Cardiac/SD/Mitral Valve/MV E Velocity':              ('E_only', 1),
    'Cardiac/SD/Mitral Valve/MV A Velocity':              ('A_only', 1),
}


def corrected(vel, nyq, bf, rbf, kind=None):
    """Spektral kaliper hızını izin gerçek baseline'ına göre düzelt (m/s, işaretli).

    YALNIZ CW (2026-09-27 düzeltmesi): PW izlerinin ~%9'unda da bf != rbf, fakat PW kaliperleri
    export'ta zaten doğru — piksel kenar testinde ham konum zarf kenarında (E/A %72, e' %91),
    düzeltilmiş konum değil. Önceki sürüm düzeltmeyi PW'ye de uyguluyordu (analiz planı §5.2'den sapma).
    """
    if kind != 'continuous_wave' or None in (nyq, bf, rbf):
        return vel
    return vel + (rbf - bf) * 2.0 * nyq


def scan_exams(pattern=EXAM_TARS):
    """Her muayene için ham ölçüm listeleri + kayıt envanteri."""
    exams = {}
    for path in sorted(glob.glob(pattern)):
        exam_id = os.path.basename(path)[:-4]
        raw = defaultdict(list)
        n_bmode = 0
        frame_dt, rr_cv = [], []
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.name.endswith('.zarr/.zattrs'):
                    continue
                d = json.load(tf.extractfile(m))
                man = d.get('recording_manifest', {})
                ct = d.get('content_types') or []
                if ct == ['2d_brightness_mode']:
                    n_bmode += 1
                    dt = (d.get('median_delta_time') or {}).get('2d_brightness_mode')
                    if dt:
                        frame_dt.append(dt)
                qrs = ((man.get('ecg') or {}).get('qrs_trigger_times')) or []
                if len(qrs) >= 4:
                    rr = [b - a for a, b in zip(qrs, qrs[1:])]
                    if st.mean(rr) > 0:
                        rr_cv.append(st.pstdev(rr) / st.mean(rr))
                for tr in (man.get('tracks') or []):
                    nyq, bf, rbf = tr.get('nyquist_mps'), tr.get('baseline_frac'), tr.get('spectral_row_baseline_frac')
                    for a in (tr.get('spectral_annotations') or []):
                        key = SPECTRAL.get(a.get('label') or '')
                        pts = a.get('points') or []
                        if not key or len(pts) != key[1]:
                            continue
                        vs = [corrected(p['velocity_mps'], nyq, bf, rbf, tr.get('semantic_id')) for p in pts]
                        raw[key[0]].append(vs)
        exams[exam_id] = {'raw': dict(raw), 'n_bmode': n_bmode,
                          'frame_dt': frame_dt, 'rr_cv': rr_cv}
    return exams


def med(xs):
    return st.median(xs) if xs else None


def exam_measurements(rec):
    """Muayene düzeyinde kanonik ölçümler (E,A cm/s; e' cm/s; TR m/s)."""
    raw = rec['raw']
    E = [abs(v[0]) * 100 for v in raw.get('EA_pair', [])]
    A = [abs(v[2]) * 100 for v in raw.get('EA_pair', [])]
    E += [abs(v[0]) * 100 for v in raw.get('E_only', [])]
    A += [abs(v[0]) * 100 for v in raw.get('A_only', [])]
    e_sep = [abs(v[0]) * 100 for v in raw.get('e_sep', [])]
    e_lat = [abs(v[0]) * 100 for v in raw.get('e_lat', [])]
    tr = [abs(v[0]) for v in raw.get('TRvel', [])]
    p = {'E': med(E), 'e_sep': med(e_sep), 'e_lat': med(e_lat), 'TRvel': med(tr)}
    a = med(A)
    p['EA'] = (p['E'] / a) if (p['E'] and a and a > 1.0) else None   # A~0 -> AF şüphesi
    p['A'] = a
    p['n_E'], p['n_esep'], p['n_elat'], p['n_TR'] = len(E), len(e_sep), len(e_lat), len(tr)
    return derive(p)


def main(out_csv='external/echoxflow_reference.csv'):
    exams = scan_exams()
    rows, summary = [], defaultdict(int)
    nabn = defaultdict(int)
    for exam_id, rec in sorted(exams.items()):
        p = exam_measurements(rec)
        g = grade_ase2025(p, age=None)          # EchoXFlow'da yaş yok -> yaş-bağımsız e' eşikleri
        rr = med(rec['rr_cv'])
        fps = (1.0 / med(rec['frame_dt'])) if rec['frame_dt'] else None
        rows.append({
            'exam_id': exam_id,
            'E': p.get('E'), 'A': p.get('A'), 'EA': p.get('EA'),
            'e_sep': p.get('e_sep'), 'e_lat': p.get('e_lat'), 'e_avg': p.get('e_avg'),
            'Ee_sep': ee_value(p, 'sep'), 'Ee_lat': ee_value(p, 'lat'), 'Ee_mean': ee_value(p, 'mean'),
            'TRvel': p.get('TRvel'),
            'n_primary_available': sum(int(x is not None) for x in
                                       (p.get('e_avg'), ee_value(p, 'mean') or ee_value(p, 'sep') or ee_value(p, 'lat'), p.get('TRvel'))),
            'n_abnormal': g.get('n_abnormal'),
            'lap': g.get('lap'), 'grade_full': g.get('grade'), 'reason': g.get('reason'),
            'n_bmode_recordings': rec['n_bmode'],
            'bmode_fps': round(fps, 1) if fps else None,
            'rr_cv': round(rr, 3) if rr is not None else None,
        })
        summary['exams'] += 1
        summary['lap_decidable'] += int(g.get('lap') is not None)
        if g.get('lap'):
            summary['lap_' + g['lap']] += 1
        if g.get('n_abnormal') is not None:
            nabn[g['n_abnormal']] += 1
        summary['reason_' + (g.get('reason') or g.get('grade') or 'ok')] += 0

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f'muayene: {summary["exams"]}   -> {out_csv}')
    print(f'LAP karar verilebilir : {summary["lap_decidable"]}')
    print(f'   yüksek dolum basıncı: {summary.get("lap_elevated", 0)}')
    print(f'   normal              : {summary.get("lap_normal", 0)}')
    print('n_abnormal (birincil 3 değişkenden anormal sayısı):',
          {k: nabn[k] for k in sorted(nabn)})
    inc = defaultdict(int)
    for r in rows:
        if r['lap'] is None:
            inc[r['reason'] or 'primer degisken yetersiz'] += 1
    print('karar verilemeyenler:', dict(inc))
    for col in ('E', 'A', 'EA', 'e_sep', 'e_lat', 'TRvel', 'Ee_mean'):
        xs = sorted(r[col] for r in rows if r[col] is not None)
        if xs:
            print(f'  {col:8s} n={len(xs):4d} medyan={st.median(xs):7.2f} '
                  f'IQR={xs[len(xs)//4]:.2f}-{xs[3*len(xs)//4]:.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
