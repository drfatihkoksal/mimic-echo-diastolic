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

Earlier descriptions of this work stated that the external cohort was acquired on another vendor's equipment. Both
cohorts were acquired on GE systems (MIMIC-IV-ECHO: Vivid E90, E95 and S7; EchoXFlow: Vivid E95 Ultra
Edition). The claim has been corrected to country, institution and acquisition format. No analysis changed.

---

# Post hoc analyses (November 2026)

Everything below was added after the primary results were known. Each analysis was specified in the author's
dated working log before it was run; code comments that cite an entry of that log ("Ek 12" to "Ek 30") refer to
the entries summarised here. The locked model, its Platt calibration, the decision threshold and the
indeterminate zone were not changed by any of them. Code is in [`external/posthoc/`](../external/posthoc/);
aggregate results are in `external/posthoc/posthoc_results.json` unless another file is named.

## 2026-11-24 — Net reduction in referrals was computed at the wrong threshold probability

`external/decision_curve.py` took net benefit from the nearest point of a 0.01 grid (0.13) and divided it by the
odds of the locked threshold (0.1286). Net benefit is now computed at the locked threshold itself. The reported
reductions fell by 7 to 12 per 1,000 studies (`external/results/decision_curve.md`). In the post hoc results the
net reduction in referrals is computed directly as 1,000 × [TN/n − (FN/n) × (1 − p)/p], and threshold
probabilities of 2% to 10% are taken as the range of interest.

## 2026-11-24 — Main external set for the guideline target

The algorithm reaches a decision when two primary variables are available. The prespecified guideline set
(191 examinations, 28 events) therefore contained 70 examinations without a tricuspid regurgitation velocity and
4 without E/e′; none of them had elevated LAP. The examinations in which all three primary variables were
measured are now reported as the main set (117 examinations, 28 events; AUROC 0.909, 0.846–0.961), and the
prespecified set as a sensitivity analysis (AUROC 0.924, 0.867–0.968). The endpoint hierarchy is unchanged:
elevated E/e′ remains the prespecified primary external target.

## 2026-11-24 — Values lying exactly on a guideline threshold were misclassified in the development labels

Structured measurements are stored in mixed units. After unit conversion, floating-point error left some derived
values a negligible distance from a threshold on which they should have lain exactly, and the comparison placed
them on the wrong side. Derived values are now rounded to six decimal places before comparison
(`load_internal(fix=True)` in `run_posthoc.py`). This changed 57 of 3,065 labels (training set: 39 from elevated
to not elevated and 8 from elevated to unclassified; validation set: 4; test set: 5 from elevated to not elevated
and 1 to unclassified). The test set became 437 studies with 80 events (previously 438 and 85). The model had
been trained and locked with the original labels and was not retrained. Internal results with the corrected
labels are now primary (AUROC 0.885, 0.840–0.924); results with the original labels are kept under
`ozgun_etiket` (AUROC 0.882, 0.839–0.921). No external value lay on a threshold.

## 2026-11-24 — Operating points and intervals

Sensitivity and specificity are reported at the rule-out bound of the indeterminate zone (0.1132), with studies
above it counted as referred, and all proportions carry Wilson 95% intervals. Results at the Youden threshold
(0.1286) and at a threshold giving 95% sensitivity in the validation set (0.0799, chosen post hoc) are kept in
the results file for description only.

## 2026-11-24 — Comparison with measured left atrial volume index (LAVI)

LAVI decides the label when the primary variables disagree, so the comparison of interest is in test studies
whose label was reached from the primary variables alone. A decision rule was written before the comparison
was run: the video model would be described as holding against LAVI if its AUROC was not lower, with the lower
confidence bound of the difference above −0.05, and if it ruled out no fewer studies at matched sensitivity.
Both conditions were met with the original and with the corrected labels.

| Set (corrected labels) | Comparator | Studies / events | Comparator AUROC | Video model AUROC | Difference (95% CI) |
|---|---|---|---|---|---|
| Label not dependent on LAVI | Measured LAVI | 245 / 24 | 0.820 | 0.893 | +0.073 (0.010 to 0.149) |
| Label not dependent on LAVI | LAVI, LV mass index, age, sex | 242 / 24 | 0.852 | 0.894 | +0.042 (−0.010 to 0.106) |
| Whole test set | Measured LAVI | 394 / 74 | 0.919 | 0.887 | −0.032 (−0.071 to 0.008) |
| Whole test set | LAVI, LV mass index, age, sex | 389 / 74 | 0.929 | 0.888 | −0.042 (−0.075 to −0.010) |

At matched sensitivity the video model ruled out 64.9% and LAVI 35.5% of the studies in the first row; the
interval of this difference included zero (−0.122 to 0.649).

## 2026-11-25 — Does the model only estimate left atrial volume?

- **Automated left atrial volume** (`panecho_la.py`). The left atrial volume output of the pretrained encoder,
  computed from the same clips without fine-tuning, had a lower AUROC than the video model in every set:
  internal test 0.783 against 0.885 (difference +0.102, 0.059 to 0.145); external main set 0.833 against 0.909
  (+0.077, 0.035 to 0.128).
- **Auxiliary head** (`grading/train_binary.py --aux-drop`, `--stop-after`; `infer_ablation.py`). The model was
  retrained with three seeds as originally trained, without LAVI in the auxiliary head, and without the
  auxiliary head (nine runs; one reproduces the locked model exactly). Removing LAVI changed the mean AUROC by
  −0.002 internally and +0.003 in the external main set; removing the head by −0.002 and −0.010. The locked
  model was not replaced.

## 2026-11-25 — Studies outside the guideline cohort

The locked model was applied to the 4,074 development-site studies outside the analysed cohort
(`a4_pipeline.py`); 4,005 had a usable clip. Of 2,765 studies to which the algorithm does not apply, 11.9% fell
in the rule-out zone and 69.4% in the rule-in zone. In the 1,265 studies the algorithm could not decide, the
output was spread across the three zones. All 44 studies with elevated LAP that an earlier selection step had
removed for missing E/A lay above the rule-out bound. The guideline provides no reference label for the first
two groups, so only the distribution of the output is reported.

## 2026-11-25 — Invasive pressure is not available for validation

`a11_wedge_count.py` counts pulmonary capillary wedge pressure measurements recorded within 24 hours of an
echocardiogram: 54 of 7,171 transthoracic studies and 2 of the 3,065 analysed studies
(`a11_wedge_count.json`). Invasive validation was therefore not attempted.

## 2026-11-25 — Alternative definition of elevated E/e′

The prespecified definition counts E/e′ as elevated when the septal, the lateral or the average value exceeds
its cutoff. With an average E/e′ of 14 or more only, in the 257 external examinations with both septal and
lateral e′ (31 events), the AUROC was 0.904 (0.848–0.951).

## 2026-11-25 — Review of the external reference standard

The review was performed by the author alone; a second reader did not take part. The panels of the two rounds
differed: in the second round the height of each marker was scaled to the calipers of the same recording, so
that round assessed the structure sampled and the timing of the wave but not the velocity. Summary of the
second round by measurement type (`qc_yazar_ozet.json`): 103 assessment units, 95 correct, 5 incorrect (3 mitral
inflow, 2 tricuspid regurgitation), 2 uncertain, 1 not assessed.

## 2026-11-25 — Checks of the external measurements against the stored spectra

- **Dataset version.** The analysis used the first public export of EchoXFlow (repository revision
  `d907c536cae16750b9e46f269a27befd4979ebdd`). The dataset was re-released as version 1.0.0 on 18 October 2026.
- **Continuous-wave correction** (`release_compare.py`, `release_compare.json`). In nine examinations compared
  point by point with the later release, all 1,820 continuous-wave points equalled the exported value plus the
  offset applied here; 1,524 of 1,526 pulsed-wave points were unchanged.
- **Pulsed-wave calipers lie inside the envelope** (`pw_scale_check.py`, `pw_scale_results.json`). The ratio of
  the envelope edge (half fall in intensity) to the caliper was 1.16 (1.15–1.18) for mitral inflow in 270
  examinations and 1.23 (1.22–1.24) for tissue Doppler in 303; for tricuspid regurgitation on continuous-wave
  spectra, the control, it was 0.90 (0.88–0.92) in 82. This supersedes the statement in the entry of 2026-09-27
  that the exported pulsed-wave caliper lies on the envelope edge: that test compared two candidate positions
  and did not measure the distance to the edge.
- **Caliper-free check in both cohorts** (`tdi_scale_check.py`, `tdi_scale_results.json`). Early diastolic tissue
  velocity measured from the envelope, divided by the recorded e′, was 0.97 (0.96–0.98) in the development
  cohort (418 studies; scanner calibration from the image files) and 1.26 (1.23–1.27) in 257 external
  examinations. The first run of the development-cohort part inverted the velocity axis and measured the
  systolic wave; the prespecified validity criterion (Spearman correlation with the recorded value of at least
  0.5) caught it (0.31), and the corrected run gave 0.81. The output of the first run is kept as
  `tdi_scale_results_ilk_kosu_eksen_ters.json`. Whether the exported pulsed-wave values are too low or the
  stored spectra too wide was not resolved.
- **Rescaling scenarios** (`pw_scale_check.py scenarios`). The exported values remain the primary analysis.

| Reference values | E/e′: examinations / events | AUROC | Guideline target: examinations / events | AUROC |
|---|---|---|---|---|
| As exported | 303 / 76 | 0.829 | 117 / 28 | 0.909 |
| E, A and e′ × 1.16 | 303 / 76 | 0.829 | 122 / 24 | 0.911 |
| E and A × 1.16; e′ × 1.225 | 303 / 64 | 0.828 | 120 / 18 | 0.924 |

## 2026-11-25 — Subgroups and resampling of patients

Internal test set, guideline target, locked model and rule (`ek29` in the results file). Sex is the `gender`
field of MIMIC-IV; age was divided at 65 years, a single cut chosen in advance. The external export has no
patient-level age or sex.

| Subgroup | Studies / events | AUROC (95% CI) | Ruled out | NPV in rule-out zone |
|---|---|---|---|---|
| Women | 270 / 51 | 0.890 (0.834–0.938) | 55.9% | 0.974 |
| Men | 167 / 29 | 0.880 (0.794–0.943) | 52.1% | 0.989 |
| Age ≥65 years | 242 / 59 | 0.855 (0.794–0.909) | 40.9% | 0.980 |
| Age <65 years | 195 / 21 | 0.912 (0.838–0.972) | 71.3% | 0.978 |

The 437 test studies came from 360 patients. Resampling patients instead of studies left the intervals almost
unchanged (AUROC 0.839–0.924; difference from measured LAVI where the label did not depend on it, 0.012 to
0.151).

## 2026-11-26 — Figures and Table 1

`make_figures.py` draws the discrimination, calibration and net-reduction-in-referrals panels and the external
flow diagram from the same cohort definitions as `run_posthoc.py`, and writes the numerical values of the
panels to a spreadsheet. `table1_v3.py` regenerates the cohort table with the corrected labels
(`table1_v3.json`, `table1_v3.md`).

