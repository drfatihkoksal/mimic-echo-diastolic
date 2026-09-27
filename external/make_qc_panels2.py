#!/usr/bin/env python3
"""
Dış validasyon — kontrol panelleri, 3. sürüm: HIZ EKSENİ YOK.

Gerekçe: veri setinin spektral metadata'sından güvenilir bir hız ekseni kurulamadı.
İki deneme de tutmadı (önce 2*nyquist/N varsayımı, sonra kaliperlerden fit edilen tek
katsayı) ve artık hatası kayıttan kayıta değişiyor — yani ölçek sabit bir metadata
ilişkisiyle ifade edilemiyor. Bu yüzden işaretin DİKEY konumunu iddia etmiyoruz.

Onun yerine, referans standardı için asıl kritik olan iki soru eksenden bağımsız
malzemeyle soruluyor (105 spektral kaydın hepsinde mevcut):
  1) HANGİ YAPI  -> 2D B-mode karesi + örnek hacim (gate) konumu çizilir.
                    Septal mi lateral anulus mü, gate mitral mi triküspit mi?
  2) HANGİ DALGA -> Kaliperlerin ZAMAN koordinatı (bu koordinat güvenilir) spektral
                    izin üstüne dikey çizgi olarak, altına da hizalı EKG şeridi ve
                    QRS tetikleri konur. E erken diyastolde mi, A QRS'in hemen
                    öncesinde mi, TR sistolde mi?
Değerler sayı olarak yazılır; makuliyet yargısı oradan verilir.
"""
from __future__ import annotations
import base64, csv, json, os, sys, tarfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'echoxflow-code', 'src'))
import cv2, zarr                                                    # noqa: E402
from echoxflow.scan.geometry import sector_geometry_from_mapping, physical_points_to_pixels, CartesianGrid  # noqa: E402
from echoxflow.scan.conversion import _sector_sample_coordinates    # noqa: E402

BM, SPW, SPH, ECGH = 240, 480, 186, 54       # B-mode karesi / spektrum / EKG şeridi
BLUE, GREEN, AMBER = (250, 120, 60), (110, 220, 140), (60, 170, 250)

KEEP = {'MV E/A': ('Mitral giriş akımı (PW)', ['E', None, 'A']),
        'Eprime Septal': ('Septal anulus doku Doppler (PW)', ["e'"]),
        'Eprime Lateral': ('Lateral anulus doku Doppler (PW)', ["e'"]),
        'TR Vmax': ('Triküspit yetersizlik jeti (CW)', ['TR'])}


def edge_row(seg, zero, direction, frac=0.20):
    """Zarfın dış kenarı: modal yoğunluğun %20'sine düşülen en dış satır."""
    N = len(seg); guard = max(4, int(0.035 * N)); idx = np.arange(N)
    side = (idx > zero + guard) if direction > 0 else (idx < zero - guard)
    if side.sum() < 8:
        return None
    prof = np.convolve(seg, np.ones(3) / 3, mode='same'); vals = prof[side]
    bg = np.percentile(vals, 10); env = vals.max()
    if env - bg < 8:
        return None
    rows = idx[side][vals > bg + frac * (env - bg)]
    return (rows.max() if direction > 0 else rows.min()) if len(rows) else None


def anchor_scale(mat, ts, anns, zero, offset):
    """Hız ölçeğini BU KAYDIN kendi kaliperlerinden sabitle.
    Metadata'dan güvenilir bir ölçek çıkmıyor (kayıtlar arası değişiyor), ama kayıt
    içinde tutarlı: aynı ölçümün atımları arasında ima edilen ölçeğin değişkenliği
    medyan %3,4 (IQR 1,7-7,0; 84 ölçüm). Bu yüzden medyanla sabitlemek güvenli.
    Döner: (m/s per satır, atımlar arası CV) — ikisi de None olabilir."""
    sc = []; names = []
    for a in anns:
        k, meta = kind_of(a.get('label'))
        if not k:
            continue
        for i, p in enumerate(a['points']):
            if i >= len(meta[1]) or meta[1][i] is None:
                continue
            v = p['velocity_mps'] + offset
            col = int(np.argmin(np.abs(ts - p['time_s'])))
            seg = mat[max(0, col - 3):col + 4].mean(axis=0)
            r = edge_row(seg, zero, +1 if v < 0 else -1)
            if r is None:
                continue
            d = abs(zero - r)
            if d >= 4:
                sc.append(abs(v) / d); names.append(meta[1][i])
    if not sc:
        return None, None
    # MV panelinde çapayı YALNIZ E kaliperlerinden al: A dalgası zarfı küçük ve
    # gürültülü, medyanı aşağı çekiyordu. Kardiyolog değerlendirmesinde sorunlu
    # bulunan MV panellerinde çapa sapması sağlamların iki katıydı (%17,1 vs %8,9).
    pref = [x for x, nm in zip(sc, names) if nm == 'E']
    use = pref if len(pref) >= 2 else sc
    s = float(np.median(use))
    cv = float(np.std(use) / np.mean(use)) if len(use) > 1 else None
    return s, cv


def kind_of(lab):
    for k, v in KEEP.items():
        if k in (lab or ''):
            return k, v
    return None, None


def bmode_with_gate(store, man):
    """B-mode karesini kartezyene çevir, ışın hattını ve örnek hacmi işaretle."""
    sec = next((s for s in man.get('sectors') or [] if 'bmode' in (s.get('semantic_id') or '')), None)
    if not sec:
        return None
    g = sector_geometry_from_mapping(sec['geometry'])
    frames = np.asarray(store[sec['frames']['zarr_path']])
    frame = frames[len(frames) // 2] if frames.ndim == 3 else frames
    grid = CartesianGrid.from_sector_height(g, BM)
    rows, cols, mask = _sector_sample_coordinates(g, grid, source_shape=frame.shape[:2])
    img = cv2.remap(frame, cols.astype(np.float32), rows.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = np.where(mask, img, 0)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gate = man.get('sampling_gate_metadata') or sec.get('sampling_gate_metadata')
    if gate:
        d = float(gate.get('gate_center_depth_m', 0)); a = float(gate.get('gate_tilt_rad', 0))
        sv = float(gate.get('gate_sample_volume_m', 0.004))
        pts = np.array([[r * np.sin(a), r * np.cos(a)] for r in
                        (g.depth_start_m, g.depth_end_m, d - sv / 2, d + sv / 2)])
        px = physical_points_to_pixels(pts, grid)
        cv2.line(rgb, tuple(px[0].astype(int)), tuple(px[1].astype(int)), (70, 70, 70), 1)
        p0, p1 = px[2].astype(int), px[3].astype(int)
        n = np.array([np.cos(a), -np.sin(a)]) * 9
        for p in (p0, p1):
            cv2.line(rgb, tuple((p - n).astype(int)), tuple((p + n).astype(int)), GREEN, 2)
        cv2.line(rgb, tuple(p0), tuple(p1), GREEN, 1)
        cv2.putText(rgb, 'gate', (int(px[3][0]) + 12, int(px[3][1]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1)
    # kareye getir
    h, w = rgb.shape[:2]
    if w != BM:
        if w > BM:
            o = (w - BM) // 2; rgb = rgb[:, o:o + BM]
        else:
            pad = np.zeros((h, BM, 3), np.uint8); o = (BM - w) // 2
            pad[:, o:o + w] = rgb; rgb = pad
    return rgb


def spectrum_strip(mat, ts, anns, ecg_sig, ecg_ts, qrs, offset=0.0, zero=None, scale=None):
    """Spektrum + hizalı EKG; yalnızca ZAMAN işaretleri (hız iddiası yok)."""
    t0, t1 = float(ts[0]), float(ts[-1])
    img = np.clip(mat.T, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (SPW, SPH), interpolation=cv2.INTER_AREA)
    top = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    out = []
    for a in anns:
        k, meta = kind_of(a.get('label'))
        if not k:
            continue
        for i, p in enumerate(a['points']):
            nm = meta[1][i] if i < len(meta[1]) else None
            if nm is None:
                continue
            x = int((p['time_s'] - t0) / max(t1 - t0, 1e-9) * (SPW - 1))
            v = p['velocity_mps'] + offset
            cv2.line(top, (x, 12), (x, SPH - 1), (110, 60, 30), 1)
            cv2.putText(top, nm, (x + 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLUE, 1)
            drawn = False
            if scale and zero is not None:
                sy = SPH / mat.shape[1]
                y = int((zero - v / scale) * sy)
                if 2 <= y < SPH - 1:
                    cv2.line(top, (x - 22, y), (x + 22, y), BLUE, 2)
                    cv2.circle(top, (x, y), 4, BLUE, -1)
                    drawn = True
            out.append({'name': nm, 'time_s': round(p['time_s'], 3),
                        'v': round(abs(v), 3), 'height_drawn': drawn,
                        'corrected': abs(offset) > 1e-6})

    strip = np.zeros((ECGH, SPW, 3), np.uint8)
    if ecg_sig is not None and len(ecg_sig) > 4:
        m = (ecg_ts >= t0) & (ecg_ts <= t1)
        if m.sum() > 4:
            sg = ecg_sig[m].astype(np.float32); tt = ecg_ts[m]
            lo, hi = np.percentile(sg, 1), np.percentile(sg, 99)
            y = np.clip((sg - lo) / max(hi - lo, 1e-6), 0, 1)
            xs = ((tt - t0) / max(t1 - t0, 1e-9) * (SPW - 1)).astype(np.int32)
            ys = (ECGH - 6 - y * (ECGH - 12)).astype(np.int32)
            cv2.polylines(strip, [np.stack([xs, ys], 1)], False, (200, 200, 200), 1)
    for q in (qrs or []):
        if t0 <= q <= t1:
            x = int((q - t0) / max(t1 - t0, 1e-9) * (SPW - 1))
            cv2.line(strip, (x, 0), (x, ECGH), AMBER, 1)
    cv2.putText(strip, 'EKG', (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return np.vstack([top, strip]), out


def main(plan_path, out_json):
    plan = json.load(open(plan_path))
    ref = {r['exam_id']: r for r in csv.DictReader(open('external/echoxflow_reference.csv'))}
    byexam = {}
    for p in plan['plan']:
        byexam.setdefault(p['exam_id'], []).append(p)
    tmp = os.path.expanduser('~/echoxflow-render/_qc')
    cards = []
    for n, (ex, recs) in enumerate(sorted(byexam.items()), 1):
        want = tuple(f"{p['rid']}.zarr/" for p in recs)
        dest = os.path.join(tmp, ex)
        if not os.path.isdir(dest):
            with tarfile.open(f'echoxflow/exams/{ex}.tar') as tf:
                tf.extractall(dest, members=[m for m in tf if any(w in m.name for w in want)], filter='data')
        panels = []
        for p in recs:
            zp = os.path.join(dest, 'exams', ex, f"{p['rid']}.zarr")
            att = json.load(open(zp + '/.zattrs')); man = att['recording_manifest']
            z = zarr.open(zp, mode='r')
            ecg = man.get('ecg') or {}
            try:
                esig = np.asarray(z[ecg['signal']['zarr_path']]).ravel() if ecg.get('signal') else None
                ets = np.asarray(z[ecg['timestamps']['zarr_path']]).ravel() if ecg.get('timestamps') else None
            except Exception:
                esig = ets = None
            bm = bmode_with_gate(z, man)
            for t in man.get('tracks') or []:
                anns = [a for a in (t.get('spectral_annotations') or []) if kind_of(a.get('label'))[0]]
                if not anns:
                    continue
                mat = np.asarray(z[t['data']['zarr_path']])
                ts = np.asarray(z[t['timestamps']['zarr_path']])
                # ofset YALNIZ CW'ye uygulanır; PW kaliperleri export'ta doğru (analysis_log Ek 7, 2026-09-27)
                off = ((t['spectral_row_baseline_frac'] - t['baseline_frac']) * 2 * t['nyquist_mps']
                       if t.get('semantic_id') == 'continuous_wave' else 0.0)
                # düzeltme değeri SIFIRA doğru kaydırır: negatif hızda +ofset, pozitifte -ofset
                sgn = 1.0 if any(q['velocity_mps'] < 0 for a in anns for q in a['points']) else -1.0
                zero = t['spectral_row_baseline_frac'] * mat.shape[1]
                sc, cv_ = anchor_scale(mat, ts, anns, zero, sgn * abs(off))
                # çapa kararsızsa yüksekliği hiç çizme (yanıltıcı olur)
                if cv_ is not None and cv_ > 0.15:
                    sc = None
                strip, cal = spectrum_strip(mat, ts, anns, esig, ets,
                                            ecg.get('qrs_trigger_times'), offset=sgn * abs(off),
                                            zero=zero, scale=sc)
                left = bm if bm is not None else np.zeros((BM, BM, 3), np.uint8)
                left = cv2.resize(left, (BM, strip.shape[0]), interpolation=cv2.INTER_AREA)
                comp = np.hstack([left, strip])
                ok, buf = cv2.imencode('.jpg', comp, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
                k, meta = kind_of(anns[0]['label'])
                gate = man.get('sampling_gate_metadata') or {}
                panels.append({'kind': k, 'title': meta[0], 'img': base64.b64encode(buf).decode(),
                               'calipers': cal, 'is_cw': t.get('semantic_id') == 'continuous_wave',
                               'meta': {'anchor_cv': (None if cv_ is None else round(100 * cv_, 1)),
                                        'sweep_s': round(float(ts[-1]) - float(ts[0]), 2),
                                        'gate_depth_cm': round(100 * float(gate.get('gate_center_depth_m', 0)), 1),
                                        'beats': len(ecg.get('qrs_trigger_times') or [])}})
        r = ref[ex]
        def g(k):
            v = r[k]
            return None if v in ('', 'None') else round(float(v), 2)
        cards.append({'exam_id': ex, 'lap': r['lap'], 'n_abnormal': r['n_abnormal'],
                      'E': g('E'), 'A': g('A'), 'EA': g('EA'), 'e_sep': g('e_sep'), 'e_lat': g('e_lat'),
                      'Ee_mean': g('Ee_mean'), 'Ee_sep': g('Ee_sep'), 'Ee_lat': g('Ee_lat'),
                      'TRvel': g('TRvel'), 'fps': g('bmode_fps'),
                      'n_bmode': int(r['n_bmode_recordings']), 'panels': panels})
        print(f'[{n}/{len(byexam)}] {ex}: {len(panels)} panel', flush=True)
    json.dump(cards, open(out_json, 'w'))
    print(f'{len(cards)} kart -> {out_json} ({os.path.getsize(out_json)/1e6:.1f} MB)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
