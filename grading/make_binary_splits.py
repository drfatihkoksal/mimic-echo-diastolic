#!/usr/bin/env python3
"""
Binary (elevated vs non-elevated left atrial pressure) split file for train_binary.py.

Carries the patient-level splits of make_splits.py over unchanged, so that the pretrained encoder has never
seen a validation or test patient, and replaces the four-class grade by the left-atrial-pressure node of the
2025 ASE algorithm (`lap_2025` from diastolic_grading.py). The four-class grade is kept as `grade_4class`.

Input : ~/mimic-echo/splits.csv, ~/mimic-echo/diastolic_labels.csv
Output: ~/mimic-echo/splits_binary.csv
"""
import csv, os

H = os.path.expanduser('~/mimic-echo/')
lab = {r['study_id']: r['lap_2025'] for r in csv.DictReader(open(H + 'diastolic_labels.csv'))}
rows = []
for r in csv.DictReader(open(H + 'splits.csv')):
    rows.append({'study_id': r['study_id'], 'subject_id': r['subject_id'], 'grade_2025': lab[r['study_id']],
                 'grade_4class': r['grade_2025'], 'confidence': r['confidence'], 'split': r['split'],
                 'n_clips': r['n_clips']})
out = os.environ.get('SPLITS_BINARY_OUT', H + 'splits_binary.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} studies -> {out}')
