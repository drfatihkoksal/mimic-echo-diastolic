# Kalibrasyon analizi

Platt kalibratörü yalnız **iç validation** setinde fit edildi (a=1,7397, b=−1,2923) ve test ile dış kohorta değiştirilmeden uygulandı. Dış sütunlar bu nedenle "kalibrasyon kurumlar arasında taşınıyor mu" sorusunu ölçer.

| kohort | n | prevalans | Brier (ham → kalibre) | beceri | eğim [%95 GA] | büyük ölçekte kesişim [%95 GA] | ort. fark |
|---|---|---|---|---|---|---|---|
| iç test — ASE LAP | 438 | %19.4 | 0.1462 → **0.0970** | +0.380 | 1.12 [0.905, 1.401] | +0.02 [-0.28, 0.281] | +0.002 |
| dış — E/e′ (ön-tanımlı birincil) | 303 | %25.1 | 0.1625 → **0.1437** | +0.235 | 0.70 [0.552, 0.891] | +0.29 [-0.103, 0.669] | +0.031 |
| dış — ASE LAP | 191 | %14.7 | 0.1273 → **0.0744** | +0.406 | 1.01 [0.762, 1.476] | -0.67 [-1.224, -0.156] | -0.050 |

## Dilim tabloları (beklenen vs gözlenen, Wilson %95 GA)

### iç test — ASE LAP

| dilim n | beklenen | gözlenen [%95 GA] |
|---|---|---|
| 44 | 0.008 | 0.023 [0.004–0.118] |
| 44 | 0.020 | 0.000 [0.000–0.080] |
| 44 | 0.038 | 0.045 [0.013–0.151] |
| 43 | 0.054 | 0.046 [0.013–0.155] |
| 44 | 0.076 | 0.023 [0.004–0.118] |
| 44 | 0.115 | 0.045 [0.013–0.151] |
| 43 | 0.166 | 0.233 [0.132–0.377] |
| 44 | 0.263 | 0.250 [0.146–0.394] |
| 44 | 0.424 | 0.477 [0.338–0.621] |
| 44 | 0.752 | 0.795 [0.655–0.888] |

### dış — E/e′ (ön-tanımlı birincil)

| dilim n | beklenen | gözlenen [%95 GA] |
|---|---|---|
| 61 | 0.010 | 0.033 [0.009–0.112] |
| 60 | 0.034 | 0.050 [0.017–0.137] |
| 61 | 0.099 | 0.147 [0.080–0.257] |
| 60 | 0.262 | 0.450 [0.331–0.575] |
| 61 | 0.692 | 0.574 [0.449–0.690] |

*İkincil (birincil sonuçlara uygulanmaz):* dış kohortta yalnız kesişim +0.29 kadar güncellenirse Brier 0.1437 → 0.1435, beceri +0.235 → +0.236.

### dış — ASE LAP

| dilim n | beklenen | gözlenen [%95 GA] |
|---|---|---|
| 38 | 0.008 | 0.000 [0.000–0.092] |
| 38 | 0.020 | 0.026 [0.005–0.135] |
| 39 | 0.058 | 0.026 [0.004–0.132] |
| 37 | 0.183 | 0.135 [0.059–0.280] |
| 39 | 0.703 | 0.538 [0.386–0.684] |

*İkincil (birincil sonuçlara uygulanmaz):* dış kohortta yalnız kesişim -0.67 kadar güncellenirse Brier 0.0744 → 0.0669, beceri +0.406 → +0.465.

