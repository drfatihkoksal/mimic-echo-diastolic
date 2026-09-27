# Availability of 2025 ASE algorithm variables, by cohort

Development cohort: 3065 studies. External cohort: 303 examinations (those evaluable against elevated E/e′).

| Variable | Role in the algorithm | Development | External |
|---|---|---|---|
| Septal e′ | primary — reduced e′ | 3026 (99%) | 283 (93%) |
| Lateral e′ | primary — reduced e′ | 3049 (99%) | 277 (91%) |
| E/e′, average | primary — increased E/e′ | 3043 (99%) | 257 (85%) |
| E/e′, septal | primary, alternative site | 3019 (98%) | 283 (93%) |
| E/e′, lateral | primary, alternative site | 3041 (99%) | 277 (91%) |
| TR velocity | primary — increased TR Vmax | 2530 (83%) | 190 (63%) |
| LA volume index | secondary — discordant primaries | 2698 (88%) | not recorded |
| Mitral E/A | branch — isolated reduced e′ | 3018 (98%) | 302 (100%) |
| Deceleration time | supportive | 3033 (99%) | not recorded |
| LVEF | context and scope | 1826 (60%) | not recorded |
| Pulmonary vein S/D | secondary, alternative | not recorded | traced in 4 examinations |
| LA reservoir strain | secondary, alternative | not recorded | 11 recordings in the full dataset |
| IVRT | secondary, sub-branch | not recorded | not recorded |

## Substitutions and decision rules

- **Development cohort.** All three primary variables and the secondary variable are available and
  the algorithm is applied as published. Studies in which it does not resolve were not labelled
  (n = 1,265 before the imaging requirement was applied).
- **External cohort.** The secondary variable is unavailable: the dataset contains neither left
  atrial volume nor body surface area, and the alternative secondary variables — pulmonary vein S/D
  and left atrial reservoir strain — are present in too few examinations to substitute. **No
  substitution was made.** The published algorithm was applied unmodified and reported only where it
  resolves without the secondary variable: where all three primary variables agree, or where reduced
  e′ is isolated and E/A ≤ 0.8. This is why 218 of 666 examinations yielded a determinate guideline
  classification, and why the second reference standard — elevated E/e′, which requires only
  pulsed-wave measurements — is available in more examinations.
- **Rhythm.** The development cohort carries rhythm information. The external cohort does not, so
  guideline scope was enforced from the recordings themselves: RR-interval variability derived from
  the ECG QRS triggers, and a measured A-wave velocity below 5 cm/s.
- **Valve scope.** Exclusions for significant mitral regurgitation, mitral stenosis and mitral
  annular calcification were applied in the development cohort. The external dataset carries no valve
  grading, so they could not be applied there; the direction of that limitation is conservative
  (see Discussion).
- **Data-quality correction.** Continuous-wave velocities in the external dataset required a
  documented baseline correction; pulsed-wave measurements, which supply E, A and e′, did not. The two
  baseline fields also differ in about 9% of pulsed-wave recordings, but there the exported calipers
  already lie on the envelope edge of the stored spectrum (pixel-level test), so no correction applies.
  An earlier version of the code corrected those recordings too, contrary to the analysis plan; this
  was fixed and is reported as a deviation.

