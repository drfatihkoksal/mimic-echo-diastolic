#!/usr/bin/env python3
"""
Karar eğrisi analizi (Vickers & Elkin 2006).

Soru: klinisyen "yanlış pozitifi yanlış negatife göre kaç kat daha az maliyetli
buluyorum" dediğinde (eşik olasılığı p_t), modeli kullanmak iki varsayılan stratejiden
üstün mü?
  net fayda = TP/n - (FP/n) * p_t/(1-p_t)
  herkesi gönder : prevalans - (1-prevalans) * p_t/(1-p_t)
  kimseyi gönderme: 0

Burada "gönder" = tam Doppler protokolüne yönlendir / yüksek dolum basıncı lehine
davran. Eşik olasılığı ekseninin anlamlı olması kalibre olasılık gerektirir; bu yüzden
Platt kalibrasyonu (yalnız iç val'de fit) uygulanmış olasılıklar kullanılır.
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = np.round(np.arange(0.02, 0.71, 0.01), 3)
SHOW = [0.05, 0.10, 0.1286, 0.15, 0.20, 0.30, 0.40, 0.50]


def net_benefit(y, p, pt):
    n = len(y); w = pt / (1 - pt)
    pred = p >= pt
    tp = float(((pred) & (y == 1)).sum()); fp = float(((pred) & (y == 0)).sum())
    return tp / n - (fp / n) * w


def curve(y, p):
    prev = float(y.mean())
    rows = []
    for pt in GRID:
        w = pt / (1 - pt)
        rows.append({'pt': float(pt), 'model': net_benefit(y, p, pt),
                     'herkes': prev - (1 - prev) * w, 'kimse': 0.0})
    return rows, prev


def cohorts():
    it = list(csv.DictReader(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/test_predictions.csv'))))
    yield ('iç test — ASE LAP',
           np.array([int(r['y_true']) for r in it]),
           np.array([float(r['p_elevated']) for r in it]))
    ex_all = list(csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv'))))
    # birincil kohort canonical_results ile aynı: E/e′ etiketi var ve AF dışlanmamış (2026-09-27 hizalaması)
    ex = [r for r in ex_all if r['y_birincil'] != '' and r['af_excluded'] == '0']
    yield ('dış — E/e′ (ön-tanımlı birincil)',
           np.array([int(r['y_birincil']) for r in ex]),
           np.array([float(r['p_kalibre']) for r in ex]))
    ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'echoxflow_reference_v2.csv')))}
    pr = {r['exam_id']: float(r['p_kalibre']) for r in ex_all}
    y2, p2 = [], []
    for e, r in ref.items():
        if e in pr and not int(r['af_excluded']) and r['lap'] in ('elevated', 'normal'):
            y2.append(int(r['lap'] == 'elevated')); p2.append(pr[e])
    yield ('dış — ASE LAP', np.array(y2), np.array(p2))


def main():
    out, js = ['# Karar eğrisi analizi', '',
               'Net fayda = TP/n − (FP/n)·p_t/(1−p_t). "Gönder" = tam Doppler protokolüne '
               'yönlendir. Kalibre olasılıklar kullanılmıştır (Platt, yalnız iç val\'de fit).', ''], {}
    for name, y, p in cohorts():
        rows, prev = curve(y, p)
        js[name] = {'n': int(len(y)), 'prevalans': round(prev, 4), 'egri': rows}
        out += [f'## {name}  (n={len(y)}, prevalans %{100*prev:.1f})', '',
                '| eşik olasılığı | model | herkesi gönder | kimseyi gönderme | modelin üstünlüğü |',
                '|---|---|---|---|---|']
        for pt in SHOW:
            r = min(rows, key=lambda x: abs(x['pt'] - pt))
            best = max(r['herkes'], 0.0)
            out.append(f"| %{100*r['pt']:.0f} | **{r['model']:+.4f}** | {r['herkes']:+.4f} | "
                       f"0,0000 | {r['model']-best:+.4f} |")
        adv = [r['pt'] for r in rows if r['model'] > max(r['herkes'], 0.0) + 1e-9]
        js[name]['ustun_aralik'] = [min(adv), max(adv)] if adv else None
        lo_g, hi_g = float(GRID[0]), float(GRID[-1])
        if adv:
            tam = (min(adv) <= lo_g + 1e-9) and (max(adv) >= hi_g - 1e-9)
            out += ['', ('Model, incelenen tüm eşik olasılığı aralığında (%{:.0f}–%{:.0f}) '
                         'her iki varsayılana da üstün.').format(100*lo_g, 100*hi_g) if tam else
                        ('Model her iki varsayılana da yalnız **%{:.0f} eşik olasılığından '
                         'itibaren** üstün; altında "herkesi gönder" daha iyidir.').format(100*min(adv))]
        else:
            out += ['', 'Model hiçbir eşik olasılığında iki varsayılanı birden geçmiyor.']
        r = min(rows, key=lambda x: abs(x['pt'] - 0.1286))
        gain = r['model'] - max(r['herkes'], 0.0)
        w = 0.1286 / (1 - 0.1286)
        if gain > 0:
            out += [f"Kilitli eşikte (%12,9): net fayda {r['model']:+.4f}, en iyi varsayılana göre "
                    f"kazanç {gain:+.4f} — kaçırılan vaka artışı olmadan 1.000 çalışmada gereksiz "
                    f"ileri tetkikte **{1000*gain/w:.0f} azalma**."]
        else:
            out += [f"Kilitli eşikte (%12,9): net fayda {r['model']:+.4f}, buna karşılık "
                    f"'herkesi gönder' {max(r['herkes'], 0.0):+.4f}. **Bu çalışma noktasında model "
                    f"net fayda sağlamıyor** (fark {gain:+.4f}); bu kohortta prevalans yüksek "
                    f"(%{100*prev:.1f}) olduğu için düşük eşik olasılıklarında herkesi göndermek "
                    f"güçlü bir varsayılandır."]
        out.append('')
    open(os.path.join(HERE, 'decision_curve.md'), 'w').write('\n'.join(out) + '\n')
    json.dump(js, open(os.path.join(HERE, 'decision_curve.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
