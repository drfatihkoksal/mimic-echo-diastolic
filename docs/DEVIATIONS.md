# Deviations from the analysis plan

This file is the dated public record of every departure from [`analysis_plan.md`](analysis_plan.md), which
was fixed in writing before any external-validation data were analysed. Each entry states what happened,
how it was found, what was changed, and whether any reported number changed.

---

## 2026-09-27 — Continuous-wave baseline correction was also applied to pulsed-wave recordings

**Plan.** §5.2 specifies a baseline correction for continuous-wave (CW) caliper velocities and states that
pulsed-wave (PW) recordings, the source of E, A and e′, receive no correction.

**What happened.** The two baseline fields (`baseline_frac`, `spectral_row_baseline_frac`) also differ in
about 9% of PW recordings (578 of 6,473). The first version of `external/build_external_reference.py`
applied the correction to every recording in which the two fields differed, including those PW recordings.

**How it was found.** During an independent re-extraction of the external measurements. A pixel-level test
of envelope-edge contrast at each caliper showed that for PW recordings the exported caliper position lies
on the envelope edge (mitral inflow 72%, tissue Doppler 91% of calipers) and the corrected position does
not; for CW recordings the clutter band lies at `spectral_row_baseline_frac` in 3,210 of 3,211 recordings,
confirming the CW correction.

**Change.** The correction is now applied only to tracks with `semantic_id == "continuous_wave"`, which
brings the code into line with the plan. The same fix was applied to the quality-control panel generator
(`external/make_qc_panels2.py`). Every external result was regenerated; model predictions were unchanged
(maximum absolute difference 0) and only reference labels changed.

**Effect.** 85 of the examinations evaluable against elevated E/e′ had altered E or e′ values in the first
version, and 28 labels were reversed. Reported results after correction:

| Endpoint | n | Events | AUROC (95% CI) |
|---|---|---|---|
| Elevated E/e′ (primary) | 303 | 76 | 0.829 (0.775–0.878) |
| Guideline-defined elevated LAP (secondary) | 191 | 28 | 0.924 (0.867–0.968) |

The endpoint hierarchy was not changed.

## 2026-09-27 — Implementation corrections found in the same review

- **Tertile boundaries.** `external/clipping_covariate.py` recomputed the tertile boundaries of the
  spectral-truncation covariate on every run. They are now locked at the prespecified values (0.0424,
  0.0842); quantiles of the current cohort are written separately for information only.
- **Cohort alignment.** `external/calibration_analysis.py` and `external/decision_curve.py` did not apply
  the rhythm exclusion to the primary cohort. They now use the same cohort definition as
  `external/canonical_results.py`.
- **Prevalences.** `external/prevalence_transfer.py` used hard-coded external prevalences; it now reads them
  from `external/results/canonical_results.json`.

## 2026-09-27 — Prespecified sensitivity analyses that had not been computed

- **Mitral disease (plan §5.4).** Excluding examinations with a traced mitral continuous-wave, mitral
  regurgitation or PISA envelope (`external/mitral_trace_exams.py`) was specified but had not been
  implemented. It is now computed in `canonical_results.py`: elevated E/e′ AUROC 0.824 (0.763–0.879),
  18 excluded; guideline LAP 0.933 (0.882–0.973), 10 excluded.
- **Majority rule (plan §5.5).** Now produced by the single-source script: AUROC 0.811 (0.740–0.876),
  n = 190.

## 2026-09-27 — Training stopping rule, as executed

The training configuration specified early stopping with a patience of eight epochs. Training of the
reported model was stopped manually by the author after epoch 8, four epochs after the best validation
AUROC (epoch 4); the epoch-4 weights were used throughout. Model selection remained on validation AUROC
only. Earlier drafts described the rule as "three consecutive epochs"; this has been corrected.

## 2026-09-27 — Quality-control review, second round

The second blinded round (plan §7) was performed 27 days after the first, in a different fixed random
order, with the reference classification removed from the page and panels regenerated after the PW fix
above. In two examinations two tricuspid panels shared one decision key in both rounds, so agreement is
reported over 103 assessment units. Decisions of both rounds are in `external/qc/`.

## 2026-10-01 — Prespecified adaptation analyses that had not been run (plan §5.6)

**Plan.** §5.6 states that preprocessing harmonisation (histogram matching) and recalibration in the external
cohort are reported as secondary analyses, so that the contribution of domain shift can be separated from a
failure to transfer.

**What happened.** Neither analysis had been implemented. The omission was found while completing the
TRIPOD+AI checklist. Both were run on 2026-10-01, after the primary results were known.

**Method.** The plan did not specify the harmonisation method. The following single specification was
written before these analyses were run, and no other variant was evaluated:

- *Recalibration* (`canonical_results.py`, `adaptation`): logistic recalibration (intercept and slope) of the
  logit of the locked probabilities, with 10-fold stratified cross-fitting; the locked threshold is applied
  unchanged.
- *Histogram matching* (`histmatch_external.py`): the reference grey-level distribution is taken from the
  pixels inside the ultrasound sector (non-zero maximum over time) of the 16 model frames of 2,000 randomly
  selected development training clips (seed 20261001). Each external clip's in-sector distribution is mapped
  to it by cumulative-distribution matching; pixels outside the sector stay at zero. Model, calibrator,
  threshold and indeterminate zone are unchanged.

**Effect.** No previously reported number changed; results are under `ikincil_uyarlama` in
`external/results/canonical_results.json`. The zero-shot analysis remains primary.

| Endpoint | Analysis | AUROC (95% CI) | Paired difference vs zero-shot | Calibration slope | CITL |
|---|---|---|---|---|---|
| Elevated E/e′ | Recalibration | 0.825 (0.772–0.875) | — | 0.96 | −0.01 |
| Elevated E/e′ | Histogram matching | 0.854 (0.805–0.897) | +0.026 (0.006 to 0.046) | 0.85 | −0.25 |
| Guideline-defined elevated LAP | Recalibration | 0.917 (0.854–0.966) | — | 0.96 | −0.01 |
| Guideline-defined elevated LAP | Histogram matching | 0.949 (0.912–0.978) | +0.025 (0.002 to 0.057) | 1.23 | −1.17 |

## 2026-10-01 — Correction of a descriptive claim

Earlier manuscript versions described the external cohort as acquired on another vendor's equipment. Both
cohorts were acquired on GE systems (MIMIC-IV-ECHO: Vivid E90, E95 and S7; EchoXFlow: Vivid E95 Ultra
Edition). The claim has been corrected to country, institution and acquisition format. No analysis changed.
