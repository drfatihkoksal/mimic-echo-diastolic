# Prevalans transfer tablosu

Kilitli çalışma noktası (iç validation setinde belirlendi, değiştirilmeden uygulandı): eşik **0.1286**, gri bölge **0.1132 – 0.3567**.

Tutulmuş iç test setinde ölçülen özellikler: duyarlılık **0.918**, özgüllük **0.703**. Aşağıda yalnız prevalans değiştirilmiştir (Bayes); testin kendi özellikleri sabittir.

## (A) Tek eşik — 1.000 çalışma başına

| prevalans | ortam | PPD | NPD | yanlış pozitif | kaçırılan (YN) | pozitif çıkan |
|---|---|---|---|---|---|---|
| %2.0 | genel popülasyon taraması (Hakem 1 örneği) | **0.059** | 0.998 | 292 | 2 | 310 |
| %5.0 | düşük riskli ayaktan hasta | **0.140** | 0.994 | 283 | 4 | 328 |
| %10.0 | seçilmemiş ayaktan eko | **0.255** | 0.987 | 268 | 8 | 359 |
| %19.4 | MIMIC-IV-ECHO iç kohort (gözlenen) | **0.426** | 0.973 | 240 | 16 | 418 |
| %14.7 | EchoXFlow dış kohort, ASE LAP (gözlenen) | **0.347** | 0.980 | 254 | 12 | 389 |
| %25.1 | EchoXFlow dış kohort, E/e′ (gözlenen) | **0.508** | 0.962 | 223 | 21 | 453 |
| %50.0 | yüksek şüpheli, seçilmiş | **0.755** | 0.895 | 149 | 41 | 608 |

## (B) İki eşikli gri bölge — 1.000 çalışma başına

Rule-out: eşiğin altı, ileri tetkik gerekmez. Belirsiz: tam Doppler protokolüne yönlendirilir. Rule-in: yüksek dolum basıncı lehine.

| prevalans | rule-out n (NPD) | belirsiz n | rule-in n (PPD) | kaçırılan (YN) |
|---|---|---|---|---|
| %2.0 | 645 (0.998) | 262 | 93 (0.132) | 1 |
| %5.0 | 628 (0.994) | 263 | 109 (0.282) | 4 |
| %10.0 | 599 (0.988) | 266 | 135 (0.453) | 7 |
| %19.4 | 543 (0.975) | 272 | 185 (0.642) | 14 |
| %14.7 | 571 (0.982) | 269 | 160 (0.562) | 10 |
| %25.1 | 510 (0.965) | 275 | 215 (0.714) | 18 |
| %50.0 | 364 (0.903) | 289 | 347 (0.882) | 35 |

## Okunuşu

- Hakem 1 haklıdır: %2 prevalansta tek eşikle PPD 0.059, yani pozitiflerin %94'i yanlış pozitif olur. Bu model genel popülasyon taraması için uygun değildir ve öyle bir iddia edilmemektedir.
- Hedeflenen kullanım eko laboratuvarında triyajdır; oradaki prevalans %15–25 aralığındadır (iki kohortta gözlenen). O aralıkta PPD 0.35–0.51, NPD 0.96–0.98.
- Klinik değer asıl olarak DIŞLAMADADIR: gri bölge stratejisinde rule-out kolu yüksek NPD ile çalışır ve 1.000 çalışmanın 543'ini (iç kohort prevalansında) ileri tetkikten muaf tutar.
- Belirsiz kalanlar zaten yapılması gereken şeye, tam Doppler protokolüne yönlendirilir; yani bu strateji hiçbir hastayı değerlendirmesiz bırakmaz.
