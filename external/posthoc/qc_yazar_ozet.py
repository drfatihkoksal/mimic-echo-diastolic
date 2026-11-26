#!/usr/bin/env python3
"""Yazarın dış referans standardı incelemesinin (2. tur) toplu özeti -> qc_yazar_ozet.json

Girdi: external/qc_tur2_kararlar.csv (exam_id, panel, karar). Yalnız toplu sayılar yazılır.
Değerlendirme birimi = (muayene, ölçüm türü). Bir muayenede aynı türden iki panel tek birimdir;
birimin kararı panellerin en kötüsüdür (bad > maybe > ok). Kararı boş birim "değerlendirilmedi" sayılır.
Çalıştırma (proje kökünden):  python3 external/posthoc/qc_yazar_ozet.py"""
import csv, json
from collections import defaultdict
from math import sqrt

SRC, OUT = 'external/qc_tur2_kararlar.csv', 'external/posthoc/qc_yazar_ozet.json'
RANK = {'bad': 2, 'maybe': 1, 'ok': 0}
GROUP = {'MV E/A': 'mitral_akim', 'TR Vmax': 'TR', 'Eprime Septal': 'doku_Doppler', 'Eprime Lateral': 'doku_Doppler'}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


units = defaultdict(list)
for r in csv.DictReader(open(SRC)):
    units[(r['exam_id'], r['panel'])].append(r['karar'].strip())


def tally(keys):
    t = {'birim': 0, 'dogru': 0, 'yanlis': 0, 'belirsiz': 0, 'degerlendirilmedi': 0}
    for k in keys:
        ds = [d for d in units[k] if d in RANK]
        t['birim'] += 1
        if not ds:
            t['degerlendirilmedi'] += 1
            continue
        w = max(ds, key=RANK.get)
        t[{'ok': 'dogru', 'maybe': 'belirsiz', 'bad': 'yanlis'}[w]] += 1
    t['degerlendirilen'] = t['birim'] - t['degerlendirilmedi']
    t['yanlis_oran'] = round(t['yanlis'] / t['degerlendirilen'], 4) if t['degerlendirilen'] else None
    t['yanlis_ga'] = wilson(t['yanlis'], t['degerlendirilen'])
    return t


out = {'kaynak': SRC, 'muayene': len({k[0] for k in units}), 'tum': tally(list(units))}
for g in sorted(set(GROUP.values())):
    out[g] = tally([k for k in units if GROUP[k[1]] == g])
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
