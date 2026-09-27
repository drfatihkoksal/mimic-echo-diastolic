#!/usr/bin/env python3
"""
Tablo 1 — kohort özellikleri. Geliştirme kohortu (MIMIC-IV-ECHO, analiz edilen 3.065 çalışma) LAP durumuna göre;
dış kohort (EchoXFlow, birincil kohort 303 muayene) yalnız Doppler ölçümleriyle (demografi veri setinde yok).
Klinik ortam: eko zamanı bir hastane yatışı içindeyse yatan; bir YBÜ yatışı içindeyse YBÜ; değilse yatış dışı.
Çıktı: external/table1.md, external/table1.json
"""
import csv, gzip, json, os, statistics as st
from collections import defaultdict
from datetime import datetime

H = os.path.expanduser('~/mimic-echo/')
M = os.environ.get('MIMIC_IV_ROOT', os.path.expanduser('~/mimiciv/3.1')).rstrip('/') + '/'
HERE = os.path.dirname(os.path.abspath(__file__))
T = lambda s: datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')

split = {r['study_id']: r for r in csv.DictReader(open(H + 'splits_binary.csv'))}
lab = {r['study_id']: r for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
when = {r['study_id']: T(r['study_datetime']) for r in csv.DictReader(open(H + 'echo-study-list.csv'))}
subj = {split[s]['subject_id'] for s in split}
adm = defaultdict(list); icu = defaultdict(list)
for r in csv.DictReader(gzip.open(M + 'hosp/admissions.csv.gz', 'rt')):
    if r['subject_id'] in subj:
        adm[r['subject_id']].append((T(r['admittime']), T(r['dischtime']), r['race'], r['admission_type']))
for r in csv.DictReader(gzip.open(M + 'icu/icustays.csv.gz', 'rt')):
    if r['subject_id'] in subj:
        icu[r['subject_id']].append((T(r['intime']), T(r['outtime'])))

def setting(sid):
    s, t = split[sid]['subject_id'], when[sid]
    inp = [a for a in adm[s] if a[0] <= t <= a[1]]
    if any(i[0] <= t <= i[1] for i in icu[s]): return 'icu', inp
    return ('inpatient' if inp else 'not_admitted'), inp

def f(r, k):
    v = r.get(k, '')
    try: return float(v)
    except Exception: return None

groups = {'all': [], 'normal': [], 'elevated': []}
for sid, sp in split.items():
    groups['all'].append(sid); groups[sp['grade_2025']].append(sid)

def med(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return '—', 0
    q = st.quantiles(xs, n=4)
    return f'{st.median(xs):.1f} ({q[0]:.1f}–{q[2]:.1f})', len(xs)

rows = []
def add(name, fn):
    rows.append((name, {g: fn(ids) for g, ids in groups.items()}))
pct = lambda k, ids: f'{k} ({100*k/len(ids):.1f}%)'
add('Studies, n', lambda ids: str(len(ids)))
add('Patients, n', lambda ids: str(len({split[s]['subject_id'] for s in ids})))
add('Age, years', lambda ids: med([f(lab[s], 'age') for s in ids])[0])
add('Female sex', lambda ids: pct(sum(lab[s]['gender'] == 'F' for s in ids), ids))
setg = {s: setting(s) for s in split}
add('Setting: ICU stay', lambda ids: pct(sum(setg[s][0] == 'icu' for s in ids), ids))
add('Setting: hospital ward', lambda ids: pct(sum(setg[s][0] == 'inpatient' for s in ids), ids))
add('Setting: not admitted', lambda ids: pct(sum(setg[s][0] == 'not_admitted' for s in ids), ids))
add('Rhythm confirmed by contemporaneous ECG', lambda ids: pct(sum(split[s]['confidence'] == 'ecg_sinus' for s in ids), ids))
def nmed(k):
    return lambda ids: (lambda m: f'{m[0]} [n={m[1]}]')(med([f(lab[s], k) for s in ids]))
for nm, k in [('LVEF, %', 'EF'), ('E, cm/s', 'E_cms'), ('E/A', 'EA'), ("Septal e′, cm/s", 'e_sep'),
              ("Lateral e′, cm/s", 'e_lat'), ("Average E/e′", 'Ee_mean'), ('TR velocity, m/s', 'TRvel'),
              ('LA volume index, mL/m²', 'LAVi'), ('LV mass index, g/m²', 'LVMi')]:
    add(nm, nmed(k))
add('LVEF < 50%', lambda ids: (lambda xs: f'{sum(x < 50 for x in xs)} ({100*sum(x < 50 for x in xs)/len(xs):.1f}% of {len(xs)})')([f(lab[s], 'EF') for s in ids if f(lab[s], 'EF') is not None]))

# dış kohort
ref = {r['exam_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'echoxflow_reference_v2.csv')))}
ex = [x['exam_id'] for x in csv.DictReader(open(os.path.join(HERE, 'external_predictions.csv')))
      if x['y_birincil'] != '' and x['af_excluded'] == '0']
extm = {}
for nm, k in [('E, cm/s', 'E'), ('E/A', 'EA'), ("Septal e′, cm/s", 'e_sep'), ("Lateral e′, cm/s", 'e_lat'),
              ("Average E/e′", 'Ee_mean'), ('TR velocity, m/s', 'TRvel')]:
    m = med([f(ref[e], k) for e in ex]); extm[nm] = f'{m[0]} [n={m[1]}]'

out = ['# Table 1. Cohort characteristics', '',
       'Development cohort: MIMIC-IV-ECHO, analysed studies, by left atrial pressure (2025 ASE). External cohort: '
       'EchoXFlow examinations evaluable against elevated E/e′; the dataset carries no demographic or clinical '
       'data, so only Doppler measurements are shown. Continuous variables: median (IQR) [n with the value].', '',
       '| Characteristic | Development, all | Normal LAP | Elevated LAP | External (EchoXFlow) |', '|---|---|---|---|---|']
for nm, d in rows:
    out.append(f"| {nm} | {d['all']} | {d['normal']} | {d['elevated']} | "
               f"{extm.get(nm, str(len(ex)) if nm == 'Studies, n' else 'not recorded')} |")
open(os.path.join(HERE, 'table1.md'), 'w').write('\n'.join(out) + '\n')
json.dump({nm: d for nm, d in rows} | {'external': extm}, open(os.path.join(HERE, 'table1.json'), 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
