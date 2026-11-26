#!/usr/bin/env python3
"""A11 (analysis_log Ek 16): invaziv doğrulama fizibilitesi — yalnız SAYIM.
Girdi: ~/mimic-echo/rev_wedge_chartevents.csv (icu/chartevents'ten itemid 223771, 226855, 226854 = PCWP; 220060 = PA diyastolik).
Eşleştirme: aynı hastada eko study_datetime ile charttime arasında <=24 saat (ayrıca <=6 saat).
Çıktı: external/posthoc/a11_wedge_count.json (yalnız toplu sayılar)."""
import bisect, csv, json, os
from datetime import datetime
H = os.path.expanduser('~/mimic-echo/'); HERE = os.path.dirname(os.path.abspath(__file__))
PCWP = {'223771', '226855', '226854'}; PAD = {'220060'}
ev = {'pcwp': {}, 'pad': {}}; raw = {'pcwp': 0, 'pad': 0}
for r in csv.DictReader(open(H + 'rev_wedge_chartevents.csv')):
    k = 'pcwp' if r['itemid'] in PCWP else 'pad' if r['itemid'] in PAD else None
    try:
        v = float(r['valuenum'])
    except ValueError:
        continue
    if k is None or not (0 < v < 80):
        continue
    raw[k] += 1
    ev[k].setdefault(r['subject_id'], []).append(datetime.strptime(r['charttime'][:19], '%Y-%m-%d %H:%M:%S'))
for k in ev:
    for s in ev[k]:
        ev[k][s].sort()
study = {r['study_id']: (r['subject_id'], datetime.strptime(r['study_datetime'][:19], '%Y-%m-%d %H:%M:%S'))
         for r in csv.DictReader(open(H + 'echo-study-list.csv'))}
split = {r['study_id']: r['split'] for r in csv.DictReader(open(H + 'splits_binary.csv'))}
tte = [r['study_id'] for r in csv.DictReader(open(H + 'diastolic_labels.csv'))]


def near(k, sid, hours):
    subj, t = study[sid]; ts = ev[k].get(subj)
    if not ts:
        return False
    i = bisect.bisect_left(ts, t)
    return any(abs((ts[j] - t).total_seconds()) <= hours * 3600 for j in (i - 1, i) if 0 <= j < len(ts))


def count(ids):
    o = {'calisma': len(ids)}
    for k in ('pcwp', 'pad'):
        for h in (24, 6):
            m = [s for s in ids if near(k, s, h)]
            o[f'{k}_{h}saat'] = {'calisma': len(m), 'hasta': len({study[s][0] for s in m})}
    return o


res = {'gecerli_olcum_satiri': raw, 'olcumu_olan_hasta': {k: len(ev[k]) for k in ev},
       'tum_TTE_7171': count(tte), 'analiz_kohortu_3065': count(list(split)),
       'test_seti_438': count([s for s in split if split[s] == 'test']),
       'validasyon_438': count([s for s in split if split[s] == 'val']),
       'egitim_2189': count([s for s in split if split[s] == 'train'])}
json.dump(res, open(os.path.join(HERE, 'a11_wedge_count.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
