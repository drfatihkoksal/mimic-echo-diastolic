#!/usr/bin/env python3
"""
Faz C — Adım 1: sızıntısız train/val/test split (KİLİTLENİR, seed=42).

Kurallar (concept.md Faz C planı):
  - Birim = study (etiket study-level). SADECE npz_hires'ta klibi olan study'ler (eğitilebilir set).
  - GRUP = subject_id → bir hastanın TÜM study'leri aynı split'te (sızıntı yok; 407 subject çok-study'li).
  - STRATIFY = grade_2025 (Normal/Grade1/Grade2/Grade3). Grade3 n=54 → her split'te temsil kontrol edilir.
  - 70/15/15 ≈ StratifiedGroupKFold(n_splits=7): fold0=test, fold1=val, fold2-6=train.

Çıktı: ~/mimic-echo/splits.csv (study_id, subject_id, grade_2025, confidence, split)
Bir kez üretilir, DEĞİŞMEZ — tüm Faz C eğitim/değerlendirmesi buna dayanır.
"""
from __future__ import annotations
import argparse, csv, glob, os
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

GRADES = ['Normal', 'Grade1', 'Grade2', 'Grade3']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.expanduser('~/mimic-echo/manifest_study.csv'))
    ap.add_argument('--hires-dir', default=os.path.expanduser('~/mimic-echo/npz_hires'))
    ap.add_argument('--out', default=os.path.expanduser('~/mimic-echo/splits.csv'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-splits', type=int, default=7)
    args = ap.parse_args()

    # 1) eğitilebilir study'ler: npz_hires'ta ≥1 klibi olanlar + klip sayısı
    clips_per_study = defaultdict(int)
    for fp in glob.glob(os.path.join(args.hires_dir, '*.npz')):
        sid = os.path.basename(fp).split('_')[0]
        clips_per_study[sid] += 1
    trainable = set(clips_per_study)

    # 2) etiketleri oku (sadece eğitilebilir + geçerli grade)
    rows = []
    with open(args.manifest, newline='') as f:
        for r in csv.DictReader(f):
            if r['study_id'] in trainable and r['grade_2025'] in GRADES:
                rows.append(r)
    study_ids = [r['study_id'] for r in rows]
    y = np.array([GRADES.index(r['grade_2025']) for r in rows])
    groups = np.array([r['subject_id'] for r in rows])
    print(f"Eğitilebilir study: {len(rows)} | subject: {len(set(groups))} "
          f"| grade: {dict(Counter(GRADES[i] for i in y))}")

    # 3) StratifiedGroupKFold: fold0=test, fold1=val, gerisi=train
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_of = np.empty(len(rows), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(study_ids, y, groups)):
        fold_of[test_idx] = fold
    split_of = np.where(fold_of == 0, 'test', np.where(fold_of == 1, 'val', 'train'))

    # 4) yaz
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['study_id', 'subject_id', 'grade_2025', 'confidence', 'split', 'n_clips'])
        for r, sp in zip(rows, split_of):
            w.writerow([r['study_id'], r['subject_id'], r['grade_2025'],
                        r.get('confidence', ''), sp, clips_per_study[r['study_id']]])
    print(f"Yazıldı: {args.out}")

    # 5) DOĞRULAMA
    print("\n=== SIZINTI KONTROLÜ ===")
    subj_splits = defaultdict(set)
    for g, sp in zip(groups, split_of): subj_splits[g].add(sp)
    leaked = [s for s, ss in subj_splits.items() if len(ss) > 1]
    print(f"Birden fazla split'te görünen subject: {len(leaked)}  ({'SIZINTI VAR!' if leaked else 'YOK — temiz'})")

    print("\n=== DAĞILIM (study) ===")
    print(f"{'split':<7}{'toplam':>8}  " + "".join(f"{g:>9}" for g in GRADES) + f"{'klip':>10}{'subj':>8}")
    for sp in ['train', 'val', 'test']:
        mask = split_of == sp
        cnt = Counter(GRADES[i] for i in y[mask])
        n = int(mask.sum()); pct = 100 * n / len(rows)
        nclip = sum(clips_per_study[study_ids[i]] for i in np.where(mask)[0])
        nsubj = len(set(groups[mask]))
        print(f"{sp:<7}{n:>6}({pct:>3.0f}%) " + "".join(f"{cnt.get(g,0):>9}" for g in GRADES)
              + f"{nclip:>10}{nsubj:>8}")
    # Grade3 her split'te var mı?
    g3 = {sp: int(((split_of == sp) & (y == GRADES.index('Grade3'))).sum()) for sp in ['train','val','test']}
    print(f"\nGrade3 split dağılımı: {g3}  ({'TÜM SPLIT KAPSANDI' if all(v>0 for v in g3.values()) else 'UYARI: bir split Grade3 içermiyor'})")
    print(f"confidence(ecg_sinus) split dağılımı: " +
          str({sp: int(((split_of==sp) & np.array([r.get('confidence')=='ecg_sinus' for r in rows])).sum()) for sp in ['train','val','test']}))

if __name__ == '__main__':
    main()
