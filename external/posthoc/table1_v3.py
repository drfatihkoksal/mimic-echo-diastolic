#!/usr/bin/env python3
"""Tablo 1 (güncel): geliştirme kohortu SINIR DÜZELTMELİ etiketle (analysis_log Ek 13–14); dış kohort sütunları
E/e′ kümesi (303) ve kılavuz hedefinin ana kümesi (üç birincil değişken ölçülmüş, 117).
Çıktı (yalnız toplu değerler): external/posthoc/table1_v3.md, table1_v3.json
Çalıştırma (proje kökünden):  python3 external/posthoc/table1_v3.py"""
import csv, gzip, json, os, sys, statistics as st
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_posthoc as RR                                         # noqa: E402

H = os.path.expanduser('~/mimic-echo/')
M = os.environ.get('MIMIC_IV_ROOT', os.path.expanduser('~/mimiciv/3.1')).rstrip('/') + '/'
T = lambda s: datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')

IF = RR.load_internal(fix=True); X = RR.load_external()
split = {r['study_id']: r for r in csv.DictReader(open(H + 'splits_binary.csv'))}
lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
when = {r['study_id']: T(r['study_datetime']) for r in csv.DictReader(open(H + 'echo-study-list.csv'))}
subj = {split[s]['subject_id'] for s in split}
adm = defaultdict(list); icu = defaultdict(list)
for r in csv.DictReader(gzip.open(M + 'hosp/admissions.csv.gz', 'rt')):
    if r['subject_id'] in subj and r['admittime'] and r['dischtime']:
        adm[r['subject_id']].append((T(r['admittime']), T(r['dischtime'])))
for r in csv.DictReader(gzip.open(M + 'icu/icustays.csv.gz', 'rt')):
    if r['subject_id'] in subj and r['intime'] and r['outtime']:
        icu[r['subject_id']].append((T(r['intime']), T(r['outtime'])))

def setting(sid):
    s, t = split[sid]['subject_id'], when[sid]
    if any(a <= t <= b for a, b in icu[s]): return 'icu'
    return 'inpatient' if any(a <= t <= b for a, b in adm[s]) else 'not_admitted'

def f(r, k):
    try: return float(r.get(k, ''))
    except Exception: return None

groups = {'all': [], 'normal': [], 'elevated': []}
for sid, r in IF.items():
    if r['lap'] in ('normal', 'elevated'):
        groups['all'].append(sid); groups[r['lap']].append(sid)

def med(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return '—', 0
    q = st.quantiles(xs, n=4)
    return f'{st.median(xs):.1f} ({q[0]:.1f}–{q[2]:.1f})', len(xs)

rows = []
def add(name, fn): rows.append((name, {g: fn(ids) for g, ids in groups.items()}))
pct = lambda k, ids: f'{k:,} ({100 * k / len(ids):.1f}%)'
add('Studies, n', lambda ids: f'{len(ids):,}')
add('Patients, n', lambda ids: f"{len({split[s]['subject_id'] for s in ids}):,}")
add('Age, years', lambda ids: med([f(lab[s], 'age') for s in ids])[0])
add('Female sex', lambda ids: pct(sum(lab[s]['gender'] == 'F' for s in ids), ids))
setg = {s: setting(s) for s in groups['all']}
add('Setting: intensive care stay', lambda ids: pct(sum(setg[s] == 'icu' for s in ids), ids))
add('Setting: hospital ward', lambda ids: pct(sum(setg[s] == 'inpatient' for s in ids), ids))
add('Setting: not admitted', lambda ids: pct(sum(setg[s] == 'not_admitted' for s in ids), ids))
nmed = lambda k: (lambda ids: (lambda m: f'{m[0]} [{m[1]:,}]')(med([f(lab[s], k) for s in ids])))
MEAS = [('LVEF, %', 'EF', None), ('E, cm/s', 'E_cms', 'E'), ('E/A', 'EA', 'EA'), ('Septal e′, cm/s', 'e_sep', None),
        ('Lateral e′, cm/s', 'e_lat', None), ('Average E/e′', 'Ee_mean', 'Ee_mean'), ('TR velocity, m/s', 'TRvel', 'TRvel'),
        ('LAVI, mL/m²', 'LAVi', None), ('LV mass index, g/m²', 'LVMi', None)]
for nm, k, _ in MEAS:
    add(nm, nmed(k))
add('Label decided by LAVI (secondary step)', lambda ids: pct(sum(IF[s]['via_sec'] for s in ids), ids))
add('All three primary variables measured', lambda ids: pct(sum(IF[s]['n_avail'] == 3 for s in ids), ids))

# dış kohort: yalnız referans dosyasındaki ölçümler (demografi yok)
ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, '..', 'echoxflow_reference_v2.csv')))}
ext = {'ee': [r['eid'] for r in X.values() if not r['af'] and r['pred'] and r['y_ee'] is not None],
       'main': [r['eid'] for r in X.values() if not r['af'] and r['pred'] and r['y_lap'] is not None and r['n_avail'] == 3]}
extm = {}
for g, ids in ext.items():
    d = {'Studies, n': f'{len(ids):,}'}
    for nm, k in [('E, cm/s', 'E'), ('E/A', 'EA'), ('Septal e′, cm/s', 'e_sep'), ('Lateral e′, cm/s', 'e_lat'),
                  ('Average E/e′', 'Ee_mean'), ('TR velocity, m/s', 'TRvel')]:
        m = med([f(ref[e], k) for e in ids]); d[nm] = f'{m[0]} [{m[1]:,}]'
    extm[g] = d

out = ['| Characteristic | Development, all | Not elevated | Elevated LAP | External, E/e′ set | External, guideline set |',
       '|---|---|---|---|---|---|']
for nm, d in rows:
    out.append(f"| {nm} | {d['all']} | {d['normal']} | {d['elevated']} | {extm['ee'].get(nm, 'not recorded')} | "
               f"{extm['main'].get(nm, 'not recorded')} |")
open(os.path.join(HERE, 'table1_v3.md'), 'w').write('\n'.join(out) + '\n')

# metinde kullanılan tekil değerler
def num(ids, fn): return fn(ids)
n = {g: len(ids) for g, ids in groups.items()}
summ = {'n': n, 'hasta': len({split[s]['subject_id'] for s in groups['all']}),
        'yas_medyan': {g: st.median([f(lab[s], 'age') for s in ids]) for g, ids in groups.items()},
        'kadin_oran': sum(lab[s]['gender'] == 'F' for s in groups['all']) / n['all'],
        'ortam_oran': {k: {g: sum(setg[s] == k for s in ids) / len(ids) for g, ids in groups.items()}
                       for k in ('icu', 'inpatient', 'not_admitted')}}
json.dump({'tablo': {nm: d for nm, d in rows}, 'dis': extm, 'ozet': summ},
          open(os.path.join(HERE, 'table1_v3.json'), 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out)); print(json.dumps(summ, ensure_ascii=False))
