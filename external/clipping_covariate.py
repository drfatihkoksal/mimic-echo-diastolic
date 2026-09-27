#!/usr/bin/env python3
"""
Kırpılma derecesini muayene düzeyinde SÜREKLİ KOVARYAT olarak hazırlar.

Mantık: kırpılma modelin girdisini bozmaz (model B-mode video kullanır, spektrumu
değil); bozduğu şey REFERANS STANDARDININ güvenilirliğidir. Buradan sınanabilir bir
öngörü çıkar: kırpılma arttıkça model-referans uyumu düşmeli. Düşerse, kalan
uyuşmazlığın bir kısmının model başarısızlığı değil etiket gürültüsü olduğu gösterilmiş
olur; düşmezse bu açıklama elenir.

Skor: son noktayı üreten izlerin kenar-sinyal oranlarının MEDYANI.
  - birincil son nokta (yüksek E/e'): MV E/A, e_sep, e_lat izleri
  - ikincil son nokta (ASE 2-kriterli LAP): yukarıdakiler + TR
"""
from __future__ import annotations
import collections, csv, json, os, statistics as st, sys

PRIMARY = {'MV E/A', 'e_sep', 'e_lat'}
SECONDARY = PRIMARY | {'TR'}


def per_exam_scores(quality_csv='external/spectral_quality.csv'):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(quality_csv)):
        if r['panel'] == 'ERR':
            continue
        by[r['exam_id']].setdefault(r['panel'], []).append(float(r['edge_ratio']))
    out = {}
    for ex, panels in by.items():
        def med(keys):
            vals = [v for k, vs in panels.items() if k in keys for v in vs]
            return round(st.median(vals), 4) if vals else None
        out[ex] = {'clip_primary': med(PRIMARY), 'clip_secondary': med(SECONDARY),
                   'n_traces': sum(len(v) for v in panels.values())}
    return out


def main():
    ref = list(csv.DictReader(open('external/echoxflow_reference.csv')))
    views = collections.Counter(x['study_id'] for x in
                                csv.DictReader(open(os.path.expanduser('~/echoxflow-render/_views.csv')))
                                if x['keep'] == '1')
    sc = per_exam_scores()

    def f(r, k):
        v = r[k]
        return float(v) if v not in ('', 'None') else None

    def af(r):
        rr, A = f(r, 'rr_cv'), f(r, 'A')
        return (rr is not None and rr > 0.15) or (A is not None and A < 5)

    rows = []
    for r in ref:
        ex = r['exam_id']
        s = sc.get(ex, {})
        r2 = dict(r)
        r2.update(clip_primary=s.get('clip_primary'), clip_secondary=s.get('clip_secondary'),
                  n_clip_traces=s.get('n_traces'), n_target_clips=views.get(ex, 0),
                  af_excluded=int(af(r)))
        rows.append(r2)
    out = 'external/echoxflow_reference_v2.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    coh = [r for r in rows if not r['af_excluded'] and r['n_target_clips'] >= 1
           and any(f(r, k) is not None for k in ('Ee_sep', 'Ee_lat', 'Ee_mean'))
           and r['clip_primary'] is not None]
    vals = sorted(r['clip_primary'] for r in coh)
    t1, t2 = st.quantiles(vals, n=3)
    terts = {'kohort_n': len(coh), 'tersil_sinirlari': [round(t1, 4), round(t2, 4)],
             'medyan': round(st.median(vals), 4), 'min': vals[0], 'max': vals[-1]}
    # Tersil sınırları ÖN-TANIMLI ve kilitli (analysis_plan §6: 0.042 / 0.084). Yeniden koşuda üzerine yazılmaz;
    # yalnız bilgi amaçlı olarak mevcut kohortun kantilleri ayrı dosyaya yazılır.
    kilit = 'external/clipping_tertiles.json'
    if os.path.exists(kilit):
        json.dump(terts, open('external/clipping_tertiles_bilgi.json', 'w'), indent=1, ensure_ascii=False)
    else:
        json.dump(terts, open(kilit, 'w'), indent=1, ensure_ascii=False)
    print(f'{out} yazıldı ({len(rows)} muayene)')
    print(f"birincil kohort n={len(coh)} | kırpılma skoru medyan {terts['medyan']} "
          f"(min {terts['min']}, max {terts['max']})")
    print(f"tersil sınırları (sonuçlar görülmeden sabitlendi): {terts['tersil_sinirlari']}")
    for i, (lo, hi) in enumerate([(vals[0], t1), (t1, t2), (t2, vals[-1])], 1):
        n = sum(1 for v in vals if (lo <= v <= hi if i == 1 else lo < v <= hi))
        print(f"  tersil {i}: {lo:.3f}-{hi:.3f}  n={n}")


if __name__ == '__main__':
    main()
