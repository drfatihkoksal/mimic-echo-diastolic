# Analysis plan

*Fixed in writing before any external validation data were analysed. Released with the code.*
*Study: Doppler-free estimation of elevated left ventricular filling pressure from B-mode
echocardiographic video, developed on MIMIC-IV-ECHO and validated on EchoXFlow.*

---

## 1. Task and endpoint

A single binary endpoint: **elevated versus non-elevated left atrial pressure**, as defined by the
2025 American Society of Echocardiography (ASE) algorithm. The algorithm is applied down to the node
that classifies left atrial pressure; grade assignment is not performed and ordinal grading is not
reported. No Doppler measurement enters the model at inference.

## 2. Development cohort

MIMIC-IV-ECHO. Studies are excluded where they fall outside the scope of the guideline algorithm,
which presupposes sinus rhythm and the absence of significant mitral disease: atrial fibrillation or
flutter, moderate-to-severe or severe mitral regurgitation, mitral stenosis, or moderate-to-severe
mitral annular calcification. Studies for which the algorithm returns no determinate answer are not
labelled.

Splits are at patient level with no patient appearing in more than one split. Split assignment is
carried over unchanged from the earlier development of this pipeline, so that the pretrained
encoder used here has never seen a validation or test patient.

## 3. Model selection

Selection is on **validation-set AUROC**. Training stops when that metric has not improved for a
prespecified number of epochs. No test-set or external-cohort quantity informs model selection,
architecture, hyperparameters, cohort definition, or preprocessing.

## 4. Decision parameters — fixed on validation only

Three parameters are estimated on the internal **validation** set, written to disk, and applied
unchanged to the internal test set and to the external cohort:

1. **Probability calibration.** Platt scaling. Training uses class-balanced sampling, so the
   uncalibrated model over-predicts; Platt scaling is monotone and therefore corrects probabilities
   without altering discrimination.
2. **Decision threshold.** The Youden index of the calibrated validation probabilities.
3. **Indeterminate zone.** Two further bounds, defined as the thresholds achieving ≥ 90% sensitivity
   and ≥ 90% specificity. Studies between them are reported as indeterminate rather than forced into
   a class.

## 5. External validation cohort

EchoXFlow (Akershus University Hospital, Norway; GE equipment; raw beamspace recordings). Validation
is **zero-shot**: no retraining, no threshold adjustment, no recalibration.

### 5.1 Reference standard

Doppler measurements are read from the acquiring sonographers' calipers in the recording metadata:
mitral inflow E and A, septal and lateral e′, tricuspid regurgitation velocity; aggregated per
examination by the median across beats.

The dataset contains no left atrial volume and no body surface area, so the secondary variable of the
guideline algorithm is unavailable. The published algorithm is therefore **applied unmodified and
reported only where it resolves without that variable** — where all three primary variables agree, or
where reduced e′ is isolated and E/A ≤ 0.8. No modified or simplified variant is substituted.

### 5.2 Data-quality correction to continuous-wave velocities

The two baseline fields in the continuous-wave metadata of this dataset are mutually inconsistent
(`baseline_frac` 0.1 versus `spectral_row_baseline_frac` 0.5) and the exported velocities are
displaced by a constant offset. Correction applied:

    v = v_reported + (rbf − bf) × 2 × nyquist

Validation of the correction, specified in advance as the criterion for accepting it: median
tricuspid regurgitation velocity and median aortic valve peak velocity must both fall within
physiological ranges after correction, and a pulsed-wave control measurement unaffected by the
inconsistency (right ventricular S′) must remain essentially unchanged.

Pulsed-wave recordings — the source of E, A and e′ — have consistent baseline fields and receive **no
correction**.

### 5.3 Rhythm exclusion

The dataset carries no rhythm annotation, so guideline scope is enforced from the recordings: an
examination is excluded when the coefficient of variation of RR intervals derived from the ECG QRS
triggers exceeds 0.15, or when a measured A-wave velocity is below 5 cm/s. An absent A-wave caliper
is not treated as evidence of atrial fibrillation; only a measured low value is.

### 5.4 Scope limitation that cannot be enforced

The dataset carries no valve grading, so the guideline's exclusions for significant mitral
regurgitation, mitral stenosis and mitral annular calcification **cannot be applied externally**.
This is reported as a limitation. Its direction is conservative: significant mitral regurgitation
raises transmitral E without a corresponding rise in filling pressure, adding label noise toward
false positives in the reference standard, which depresses rather than inflates measured performance.
A sensitivity analysis excludes examinations in which the sonographer traced a mitral
continuous-wave, mitral regurgitation or PISA envelope, on the grounds that mitral regurgitation is
traced when it is noticeable.

### 5.5 Reference standards reported

Both of the following are reported; they answer different questions.

- **Elevated left atrial pressure by the 2025 ASE algorithm** — the target the model was trained on.
- **Elevated E/e′** (septal ≥ 15, lateral ≥ 13, or average ≥ 14) — independent of the training
  objective, independent of the correction in §5.2, available in more examinations, and directly
  comparable with previously reported automated Doppler workflows.

A third, the majority rule among the three primary variables where all three were measured, is
reported as a sensitivity analysis.

### 5.6 Image processing

Beamspace-to-Cartesian conversion, cropping to the ultrasound sector, padding to square without
changing the aspect ratio, and resizing to 224 × 224 — the identical geometric operation applied to
the development images. View classification uses the same classifier, unchanged, in both cohorts.

The primary external analysis is pure zero-shot. Preprocessing harmonisation (histogram matching) and
recalibration in the external cohort are reported as secondary analyses, so that the contribution of
domain shift can be separated from that of genuine non-transfer.

## 6. Reference-standard quality as a covariate

The velocity scale of the external spectral recordings varies between recordings in a way the
metadata does not capture, and in some recordings the Doppler envelope is truncated at the edge of
the stored frame. This is quantified per trace as the proportion of pixels above the 60th percentile
within the outermost 3% band of the velocity axis, measured on the side where the flow is displayed
and left undefined where that band lies within 20% of the image height of the baseline. The
examination-level score is the median across the traces that produce the reference standard.

Truncation is common rather than exceptional in this dataset and is therefore analysed as a
**continuous covariate, not an exclusion criterion**. Tertile boundaries were fixed before the
external analysis at **0.042 and 0.084**.

**Prespecified prediction.** Truncation degrades the reference standard, not the model input — the
model sees only B-mode video and never these recordings. Agreement between model and reference should
therefore fall as truncation increases. If it does not, that explanation for any residual
disagreement is eliminated and the disagreement is attributed to the model. Both outcomes are
informative, which is why the prediction is stated in advance.

## 7. Quality control of the external reference standard

Thirty examinations sampled at random, stratified by reference label, reviewed by the author blinded
to the reference classification. For each Doppler panel the review displays the recording's own
two-dimensional image with the sample-volume position marked, the spectral trace with the caliper
times marked, and the aligned electrocardiogram, and asks whether each caliper lies on the correct
structure and the correct wave. A second blinded round on the same set, separated by several weeks,
provides intra-observer agreement.

## 8. Statistical analysis

- Discrimination: AUROC with 95% confidence intervals from 4000 bootstrap resamples at a fixed seed.
- Calibration: Brier score with its prevalence-determined baseline and skill score, the Murphy
  decomposition into reliability and resolution, calibration slope, and calibration-in-the-large.
- Operating characteristics at the locked threshold, with the underlying counts; predictive values
  transferred across prevalence by Bayes' theorem.
- Clinical utility: decision-curve analysis.
- **Single source.** Every reported number is produced by one script from one results file at one
  seed, and the figures read that same file, so that text and figures cannot diverge and identical
  quantities cannot carry different intervals.

## 9. Fixed

After this document was written, the following are not changed in response to results: the endpoint
definition, the reference-standard definitions, the cohort criteria and exclusion rules, the
threshold and calibration procedure, and the tertile boundaries of the quality covariate. Any
deviation is recorded, dated, with its reason, in the internal analysis log.
