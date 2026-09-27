#!/usr/bin/env python3
"""
Kalibrasyon analizi: iç test ve iki dış son nokta.

Kalibratör (Platt) YALNIZ iç validation setinde fit edilmişti ve dış kohorta
değiştirilmeden uygulandı. Buradaki soru bu yüzden anlamlı: kalibrasyon kurumlar
arasında taşınıyor mu?

Raporlanan:
  - kalibrasyon eğimi  : logit(p) üzerine lojistik regresyonun katsayısı (ideal 1)
                         <1 aşırı uç tahminler, >1 fazla temkinli tahminler
  - büyük ölçekte kalibrasyon (calibration-in-the-large): eğim 1'e sabitlenip
                         logit(p) offset alınarak kesişim (ideal 0)
  - gözlenen - beklenen ortalama fark
  - dilim tablosu (beklenen vs gözlenen, Wilson %95 GA)
İkincil (BİRİNCİL SONUÇLARA UYGULANMAZ): dış kohortta yalnız kesişim güncellemesiyle
yeniden kalibrasyonun ne kazandıracağı.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260901)


def logit(p, e=1e-6):
    p = np.clip(np.asarray(p, float), e, 1 - e)
    return np.log(p / (1 - p))


def slope(y, p):
    return float(LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y).coef_[0][0])


def citl(y, p):
    """Eğim 1'e sabit, yalnız kesişim: logit(p) offset olarak alınır."""
    z = logit(p)
    b = 0.0
    for _ in range(200):                       # Newton
        q = 1 / (1 + np.exp(-(z + b)))
        g = np.sum(y - q); h = np.sum(q * (1 - q))
        if h < 1e-9:
            break
        step = g / h; b += step
        if abs(step) < 1e-10:
            break
    return float(b)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def bins(y, p, nb):
    q = np.quantile(p, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, nb - 1)
    out = []
    for b in range(nb):
        m = idx == b
        if m.sum() < 5:
            continue
        k = int(y[m].sum()); n = int(m.sum())
        lo, hi = wilson(k, n)
        out.append({'n': n, 'beklenen': round(float(p[m].mean()), 4),
                    'gozlenen': round(k / n, 4), 'ga': [round(lo, 4), round(hi, 4)]})
    return out


def boot_stat(y, p, fn, n=2000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        try:
            v.append(fn(y[i], p[i]))
        except Exception:
            pass
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def cohorts():
    it = list(csv.DictReader(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/test_predictions.csv'))))
    yield ('iç test — ASE LAP', np.array([int(r['y_true']) for r in it]),
           np.array([float(r['p_elevated']) for r in it]),
           np.array([float(r['p_ham']) for r in it]), 10)
    ex_all = list(csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv'))))
    # birincil kohort canonical_results ile aynı: E/e′ etiketi var ve AF dışlanmamış (2026-09-27 hizalaması)
    ex = [r for r in ex_all if r['y_birincil'] != '' and r['af_excluded'] == '0']
    yield ('dış — E/e′ (ön-tanımlı birincil)', np.array([int(r['y_birincil']) for r in ex]),
           np.array([float(r['p_kalibre']) for r in ex]),
           np.array([float(r['p_ham']) for r in ex]), 5)
    ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'echoxflow_reference_v2.csv')))}
    pc = {r['exam_id']: (float(r['p_kalibre']), float(r['p_ham'])) for r in ex_all}
    y, p, ph = [], [], []
    for e, r in ref.items():
        if e in pc and not int(r['af_excluded']) and r['lap'] in ('elevated', 'normal'):
            y.append(int(r['lap'] == 'elevated')); p.append(pc[e][0]); ph.append(pc[e][1])
    yield ('dış — ASE LAP', np.array(y), np.array(p), np.array(ph), 5)


def main():
    out = ['# Kalibrasyon analizi', '',
           'Platt kalibratörü yalnız **iç validation** setinde fit edildi (a=1,7397, b=−1,2923) ve '
           'test ile dış kohorta değiştirilmeden uygulandı. Dış sütunlar bu nedenle '
           '"kalibrasyon kurumlar arasında taşınıyor mu" sorusunu ölçer.', '',
           '| kohort | n | prevalans | Brier (ham → kalibre) | beceri | eğim [%95 GA] | büyük ölçekte kesişim [%95 GA] | ort. fark |',
           '|---|---|---|---|---|---|---|---|']
    js = {}
    for name, y, p, ph, nb in cohorts():
        unc = float(y.mean() * (1 - y.mean()))
        b_raw, b_cal = brier_score_loss(y, ph), brier_score_loss(y, p)
        sl, ic = slope(y, p), citl(y, p)
        out.append(f'| {name} | {len(y)} | %{100*y.mean():.1f} | {b_raw:.4f} → **{b_cal:.4f}** | '
                   f'{1-b_cal/unc:+.3f} | {sl:.2f} {boot_stat(y,p,slope)} | '
                   f'{ic:+.2f} {boot_stat(y,p,citl)} | {y.mean()-p.mean():+.3f} |')
        js[name] = {'n': int(len(y)), 'prevalans': round(float(y.mean()), 4),
                    'brier_ham': round(b_raw, 4), 'brier_kalibre': round(b_cal, 4),
                    'beceri': round(1 - b_cal / unc, 4), 'egim': round(sl, 3),
                    'egim_ga': boot_stat(y, p, slope), 'kesisim': round(ic, 3),
                    'kesisim_ga': boot_stat(y, p, citl),
                    'ortalama_fark': round(float(y.mean() - p.mean()), 4),
                    'dilimler': bins(y, p, nb)}
        # ikincil: dış kohortta yalnız kesişim güncellemesi
        if name.startswith('dış'):
            p2 = 1 / (1 + np.exp(-(logit(p) + ic)))
            js[name]['ikincil_yeniden_kalibre'] = {
                'kesisim_guncellemesi': round(ic, 3),
                'brier': round(brier_score_loss(y, p2), 4),
                'beceri': round(1 - brier_score_loss(y, p2) / unc, 4)}
    out += ['', '## Dilim tabloları (beklenen vs gözlenen, Wilson %95 GA)', '']
    for name, d in js.items():
        out += [f'### {name}', '', '| dilim n | beklenen | gözlenen [%95 GA] |', '|---|---|---|']
        for b in d['dilimler']:
            out.append(f"| {b['n']} | {b['beklenen']:.3f} | {b['gozlenen']:.3f} "
                       f"[{b['ga'][0]:.3f}–{b['ga'][1]:.3f}] |")
        out.append('')
        if 'ikincil_yeniden_kalibre' in d:
            r = d['ikincil_yeniden_kalibre']
            out += [f"*İkincil (birincil sonuçlara uygulanmaz):* dış kohortta yalnız kesişim "
                    f"{r['kesisim_guncellemesi']:+.2f} kadar güncellenirse Brier "
                    f"{d['brier_kalibre']:.4f} → {r['brier']:.4f}, beceri "
                    f"{d['beceri']:+.3f} → {r['beceri']:+.3f}.", '']
    open(os.path.join(HERE, 'calibration.md'), 'w').write('\n'.join(out) + '\n')
    json.dump(js, open(os.path.join(HERE, 'calibration.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
