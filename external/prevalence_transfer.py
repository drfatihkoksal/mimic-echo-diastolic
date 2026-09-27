#!/usr/bin/env python3
"""
Prevalans transfer tablosu.

JASE Hakem 1: "with the reported sensitivity of 0.88 and specificity of 0.78, a
prevalence of 2% in a general-population screening setting would yield a PPV of only
approximately 8%... The authors should discuss whether such performance is acceptable
for a screening test."

İtiraz haklıdır ve cevabı testin özelliklerini değiştirmek değil, kullanım yerini
doğru tanımlamaktır. Bu betik kilitli çalışma noktasının duyarlılık/özgüllüğünü
prevalans yelpazesine taşır ve iki stratejiyi karşılaştırır:

  (A) tek eşik           : pozitif / negatif
  (B) iki eşikli gri bölge: rule-out / belirsiz (tam Doppler protokolüne yönlendir) / rule-in

Çalışma özellikleri KİLİTLİ sınırlarla, tutulmuş test setinden ölçülür; tabloya
taşınırken yalnız prevalans değiştirilir (Bayes).
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Gözlenen prevalanslar sabit yazılmaz; tek kaynak canonical_results.json'dan okunur (2026-09-27).
_K = json.load(open(os.path.join(HERE, 'canonical_results.json')))['kohortlar']
P_IC = round(_K['ic_test_ASE_LAP']['prevalans'], 3)
P_LAP = round(_K['dis_ASE_LAP']['prevalans'], 3)
P_EE = round(_K['dis_Ee_birincil']['prevalans'], 3)
P_LO, P_HI = min(P_IC, P_LAP, P_EE), max(P_IC, P_LAP, P_EE)
PREVS = [(0.02, 'genel popülasyon taraması (Hakem 1 örneği)'),
         (0.05, 'düşük riskli ayaktan hasta'),
         (0.10, 'seçilmemiş ayaktan eko'),
         (P_IC, 'MIMIC-IV-ECHO iç kohort (gözlenen)'),
         (P_LAP, 'EchoXFlow dış kohort, ASE LAP (gözlenen)'),
         (P_EE, 'EchoXFlow dış kohort, E/e′ (gözlenen)'),
         (0.50, 'yüksek şüpheli, seçilmiş')]


def rates(y, p, thr, lo, hi):
    """Tek eşik ve üç yollu strateji için koşullu oranlar."""
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = y == 1, y == 0
    r = {'sens': float((p[pos] >= thr).mean()), 'spec': float((p[neg] < thr).mean())}
    r['ruleout_p1'] = float((p[pos] <= lo).mean())      # hasta olup dışlanan (yanlış dışlama)
    r['ruleout_p0'] = float((p[neg] <= lo).mean())
    r['rulein_p1'] = float((p[pos] >= hi).mean())
    r['rulein_p0'] = float((p[neg] >= hi).mean())
    r['ind_p1'] = 1 - r['ruleout_p1'] - r['rulein_p1']
    r['ind_p0'] = 1 - r['ruleout_p0'] - r['rulein_p0']
    return r


def transfer(r, prev, n=1000):
    d, h = n * (1 - prev), n * prev
    tp, fn = h * r['sens'], h * (1 - r['sens'])
    tn, fp = d * r['spec'], d * (1 - r['spec'])
    single = {'PPD': tp / max(tp + fp, 1e-9), 'NPD': tn / max(tn + fn, 1e-9),
              'YP_1000': fp, 'YN_1000': fn, 'pozitif_1000': tp + fp}
    ro1, ro0 = h * r['ruleout_p1'], d * r['ruleout_p0']
    ri1, ri0 = h * r['rulein_p1'], d * r['rulein_p0']
    ind = h * r['ind_p1'] + d * r['ind_p0']
    gz = {'ruleout_n': ro0 + ro1, 'ruleout_NPD': ro0 / max(ro0 + ro1, 1e-9),
          'rulein_n': ri0 + ri1, 'rulein_PPD': ri1 / max(ri1 + ri0, 1e-9),
          'belirsiz_n': ind, 'kacirilan_1000': ro1}
    return single, gz


def main():
    it = list(csv.DictReader(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/test_predictions.csv'))))
    y = [int(r['y_true']) for r in it]; p = [float(r['p_elevated']) for r in it]
    L = json.load(open(os.path.expanduser('~/mimic-echo/runs/b2_binary/binary_results.json')))
    thr, lo, hi = L['esik'], L['gri_bolge']['rule_out_alti'], L['gri_bolge']['rule_in_ustu']
    r = rates(y, p, thr, lo, hi)

    out = [
        '# Prevalans transfer tablosu',
        '',
        f'Kilitli çalışma noktası (iç validation setinde belirlendi, değiştirilmeden uygulandı): '
        f'eşik **{thr:.4f}**, gri bölge **{lo:.4f} – {hi:.4f}**.',
        '',
        f'Tutulmuş iç test setinde ölçülen özellikler: duyarlılık **{r["sens"]:.3f}**, '
        f'özgüllük **{r["spec"]:.3f}**. Aşağıda yalnız prevalans değiştirilmiştir (Bayes); '
        'testin kendi özellikleri sabittir.',
        '',
        '## (A) Tek eşik — 1.000 çalışma başına',
        '',
        '| prevalans | ortam | PPD | NPD | yanlış pozitif | kaçırılan (YN) | pozitif çıkan |',
        '|---|---|---|---|---|---|---|',
    ]
    js = {'kilitli': {'esik': thr, 'gri_bolge': [lo, hi]}, 'ozellikler': r, 'satirlar': []}
    for prev, ad in PREVS:
        s, g = transfer(r, prev)
        out.append(f'| %{prev*100:.1f} | {ad} | **{s["PPD"]:.3f}** | {s["NPD"]:.3f} | '
                   f'{s["YP_1000"]:.0f} | {s["YN_1000"]:.0f} | {s["pozitif_1000"]:.0f} |')
        js['satirlar'].append({'prevalans': prev, 'ortam': ad, 'tek_esik': s, 'gri_bolge': g})
    out += ['', '## (B) İki eşikli gri bölge — 1.000 çalışma başına', '',
            'Rule-out: eşiğin altı, ileri tetkik gerekmez. Belirsiz: tam Doppler protokolüne '
            'yönlendirilir. Rule-in: yüksek dolum basıncı lehine.', '',
            '| prevalans | rule-out n (NPD) | belirsiz n | rule-in n (PPD) | kaçırılan (YN) |',
            '|---|---|---|---|---|']
    for prev, ad in PREVS:
        _, g = transfer(r, prev)
        out.append(f'| %{prev*100:.1f} | {g["ruleout_n"]:.0f} ({g["ruleout_NPD"]:.3f}) | '
                   f'{g["belirsiz_n"]:.0f} | {g["rulein_n"]:.0f} ({g["rulein_PPD"]:.3f}) | '
                   f'{g["kacirilan_1000"]:.0f} |')
    out += ['', '## Okunuşu', '',
            f'- Hakem 1 haklıdır: %2 prevalansta tek eşikle PPD '
            f'{transfer(r,0.02)[0]["PPD"]:.3f}, yani pozitiflerin '
            f'%{100*(1-transfer(r,0.02)[0]["PPD"]):.0f}\'i yanlış pozitif olur. '
            'Bu model genel popülasyon taraması için uygun değildir ve öyle bir iddia edilmemektedir.',
            f'- Hedeflenen kullanım eko laboratuvarında triyajdır; oradaki prevalans '
            f'%{100*P_LO:.0f}–{100*P_HI:.0f} aralığındadır (iki kohortta gözlenen). O aralıkta PPD '
            f'{transfer(r,P_LO)[0]["PPD"]:.2f}–{transfer(r,P_HI)[0]["PPD"]:.2f}, '
            f'NPD {transfer(r,P_HI)[0]["NPD"]:.2f}–{transfer(r,P_LO)[0]["NPD"]:.2f}.',
            '- Klinik değer asıl olarak DIŞLAMADADIR: gri bölge stratejisinde rule-out kolu '
            f'yüksek NPD ile çalışır ve 1.000 çalışmanın '
            f'{transfer(r,P_IC)[1]["ruleout_n"]:.0f}\'ini (iç kohort prevalansında) ileri tetkikten muaf tutar.',
            '- Belirsiz kalanlar zaten yapılması gereken şeye, tam Doppler protokolüne yönlendirilir; '
            'yani bu strateji hiçbir hastayı değerlendirmesiz bırakmaz.']
    md = os.path.join(HERE, 'prevalence_transfer.md')
    open(md, 'w').write('\n'.join(out) + '\n')
    json.dump(js, open(os.path.join(HERE, 'prevalence_transfer.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
