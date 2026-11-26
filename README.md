# Doppler-free triage for elevated left atrial pressure

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22994944.svg)](https://doi.org/10.5281/zenodo.22994944)

Code for estimating **elevated left atrial pressure from B-mode echocardiographic video alone — no Doppler
at inference** — developed on the public MIMIC-IV-ECHO dataset and validated, without adaptation, on the
independent public EchoXFlow dataset (Akershus University Hospital, Norway).

An earlier four-class grading version of this work is preserved in the git history (commit `816a474`).

| | Internal test (MIMIC-IV-ECHO) | External (EchoXFlow) |
|---|---|---|
| Elevated E/e′ (prespecified primary external target) | AUROC 0.824 (0.776–0.868), n = 437 | AUROC 0.829 (0.776–0.878), n = 303 |
| Guideline-defined elevated LAP | AUROC 0.885 (0.840–0.924), n = 437 | AUROC 0.909 (0.846–0.961), n = 117 |

The guideline target can be defined externally only where the primary variables alone decide the label; the
external row refers to the examinations with all three primary variables measured (28 events). In the
prespecified set of 191 examinations the AUROC was 0.924 (0.867–0.968). Internal values use labels corrected
for a threshold-comparison error (v3.0.0; see [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)); with the original
labels the internal AUROC was 0.882 (0.839–0.921), n = 438.

The prespecified analyses are produced by `external/canonical_results.py` and stored in
[`external/results/`](external/results/). The post hoc analyses added in v3.0.0 are produced by
`external/posthoc/run_posthoc.py` and stored in [`external/posthoc/`](external/posthoc/) (fixed seed, 4,000
bootstrap resamples in both). Prespecified secondary analyses
(plan §5.6, run in v2.1.0): cross-fitted recalibration in the external cohort restores calibration without
changing discrimination; histogram matching of image intensity raises the external AUROC to 0.854 (elevated
E/e′) and 0.949 (guideline LAP). The zero-shot results above remain primary; see
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

---

## Prespecification and deviations

- [`docs/analysis_plan.md`](docs/analysis_plan.md) — written before any external data were analysed:
  endpoints, reference standards, cohort rules, decision parameters, the quality covariate and its
  tertile boundaries.
- [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) — the dated record of every departure from that plan, and of
  every post hoc analysis added after the primary results were known.
- [`docs/parameter_availability.md`](docs/parameter_availability.md) — which 2025 ASE variables were
  available in each cohort, and the decision rules applied.

## ⚠️ No restricted data in this repository

MIMIC-IV-ECHO and MIMIC-IV are **credentialed-access** datasets distributed by PhysioNet under a data use
agreement that prohibits redistributing the data or patient-level data derived from it. This repository
contains **code, aggregate results and the author's quality-control decisions only** — no images, labels,
splits, per-study predictions or model weights for MIMIC. These are regenerated deterministically by the
code from the source data.

EchoXFlow is public (CC BY-NC-SA 4.0): <https://huggingface.co/datasets/Ahus-AIM/EchoXFlow>. Derived
EchoXFlow measurements are not redistributed here either; `external/build_external_reference.py`
regenerates them from the dataset's own metadata.

To run the development pipeline you need your own credentialed access to **MIMIC-IV-ECHO v1.0**
(<https://doi.org/10.13026/nrjh-5r77>), **MIMIC-IV** (<https://doi.org/10.13026/6mm1-ek67>) and
**MIMIC-IV-ECG**.

---

## Pipeline

### Development (MIMIC-IV-ECHO) — `grading/`

| Stage | Script | Purpose |
|---|---|---|
| Labels | `diastolic_grading.py` | 2025 ASE algorithm applied to structured measurements, stopping at the left-atrial-pressure node; excludes atrial fibrillation and significant mitral disease. Never imputes a label. |
| Preprocess | `shrink_pipeline.py`, `run_local.py`, `redownload_hires.py`, `build_local_clips.py` | DICOM → grayscale B-mode cine → sector crop → 224×224. |
| Views | `view_classify.py` | EchoPrime view classifier; keeps A4C / A2C / PLAX. |
| Splits | `make_splits.py`, `make_binary_splits.py` | Patient-level 70/15/15; the binary file carries the same splits with the LAP label. |
| Model | `model.py`, `dataset.py`, `train_binary.py` | PanEcho video encoder fine-tuned with a binary head and an auxiliary regression head (training signal only). |
| Locking | `finalize_binary.py` | Platt calibration, Youden threshold and indeterminate zone, fitted on the validation set only. |

```bash
python3 diastolic_grading.py
python3 shrink_pipeline.py && python3 view_classify.py
python3 make_splits.py && python3 make_binary_splits.py
python3 train_binary.py --out-dir ~/mimic-echo/runs/b2_binary
python3 finalize_binary.py --run-dir ~/mimic-echo/runs/b2_binary
```

### External validation (EchoXFlow) — `external/`

| Stage | Script |
|---|---|
| Reference standard from sonographers' calipers (CW baseline correction, CW only) | `build_external_reference.py` |
| Spectral-truncation covariate; rhythm exclusion | `spectral_quality.py`, `clipping_covariate.py` |
| Beamspace → Cartesian rendering of B-mode recordings | `render_echoxflow.py` (cohort list `render_cohort.txt`), then `grading/view_classify.py` |
| Zero-shot inference with the locked model, threshold and calibration | `run_external.py` |
| **All reported numbers (single source)** | `canonical_results.py` |
| Calibration, decision curves, prevalence transfer, structural confounding, Table 1 | `calibration_analysis.py`, `decision_curve.py`, `prevalence_transfer.py`, `structural_confounder.py`, `table1.py` |
| Mitral sensitivity analysis cohort | `mitral_trace_exams.py` |
| Intensity harmonisation (histogram matching) inference, plan §5.6 | `histmatch_external.py` |
| Figures | `make_external_figures.py` |
| Blinded reader review of the reference standard | `make_qc_panels2.py`, `build_qc_page.py`, `build_qc_page_tur2.py`; plan and decisions in `qc/` |

```bash
python3 external/build_external_reference.py
python3 external/spectral_quality.py && python3 external/clipping_covariate.py
python3 external/render_echoxflow.py && python3 grading/view_classify.py --npz-dir ~/echoxflow-render/npz
python3 external/run_external.py
python3 external/mitral_trace_exams.py
python3 external/histmatch_external.py
python3 external/canonical_results.py
python3 external/calibration_analysis.py && python3 external/decision_curve.py && python3 external/prevalence_transfer.py
python3 external/make_external_figures.py
```

### Post hoc analyses — `external/posthoc/`

| Analysis | Script | Results |
|---|---|---|
| Corrected labels, operating points, main external set, comparison with measured LAVI, subgroups, resampling of patients (single script) | `run_posthoc.py` | `posthoc_results.json` |
| Automated left atrial volume from the pretrained encoder | `panecho_la.py` | in `posthoc_results.json` |
| Retraining with and without LAVI supervision | `run_ablation_chain.sh`, `infer_ablation.py` | in `posthoc_results.json` |
| Locked model applied to studies outside the guideline cohort | `run_a4_chain.sh`, `a4_pipeline.py` | in `posthoc_results.json` |
| Availability of invasive pressure | `a11_wedge_count.py` | `a11_wedge_count.json` |
| Exported calipers against the stored Doppler envelope; rescaling scenarios | `pw_scale_check.py` | `pw_scale_results.json` |
| Caliper-free tissue-Doppler check in both cohorts | `tdi_scale_check.py` | `tdi_scale_results.json` |
| Comparison with the later release of EchoXFlow | `release_compare.py` | `release_compare.json` |
| Summary of the reference-standard review | `qc_yazar_ozet.py` | `qc_yazar_ozet.json` |
| Cohort table with corrected labels; figures | `table1_v3.py`, `make_figures.py` | `table1_v3.json`, `table1_v3.md` |

```bash
python3 external/posthoc/run_posthoc.py
python3 external/posthoc/pw_scale_check.py measure && python3 external/posthoc/pw_scale_check.py summarize
python3 external/posthoc/pw_scale_check.py scenarios
python3 external/posthoc/tdi_scale_check.py mimic && python3 external/posthoc/tdi_scale_check.py echox
python3 external/posthoc/tdi_scale_check.py summarize
python3 external/posthoc/table1_v3.py && python3 external/posthoc/make_figures.py figures/
```

Scripts run from the project root. Paths default to `~/mimic-echo/`, `~/echoxflow-render/` and
`echoxflow/`; machine-specific roots can be set with `MIMIC_IV_ROOT`, `MIMIC_ECHO_DICOM_ROOT` and
`GCP_BILLING_PROJECT`.

## Requirements

Python 3.13, PyTorch 2.11 (CUDA 12.8), scikit-learn, pydicom, OpenCV, zarr, numcodecs, matplotlib; see
`requirements.txt`. A single 24 GB GPU is sufficient. The video encoder is **PanEcho** (Holste et al.,
*JAMA* 2025), loaded from `torch.hub` (`CarDS-Yale/PanEcho`); the view classifier is from **EchoPrime**
(Vukadinovic et al., *Nature* 2025). Neither is vendored here. EchoXFlow reading and scan conversion use
the dataset's reference implementation (<https://github.com/Ahus-AIM/EchoXFlow>).

## Use of an AI coding agent

An AI coding agent wrote and executed much of this code under the author's direction. The study design, the
analysis plan and every analytical decision were the author's, and each implementation detail the plan did
not specify was reviewed and approved by the author before use.

## Citation

All archived versions: https://doi.org/10.5281/zenodo.22994944 (resolves to the latest release; v2.0.0: https://doi.org/10.5281/zenodo.22994945).
See `CITATION.cff`. Please also cite MIMIC-IV-ECHO, MIMIC-IV, PhysioNet and EchoXFlow.

## Licence

Code: MIT (see `LICENSE`). The licence covers **this code only** and confers no rights over MIMIC-IV-ECHO,
MIMIC-IV or EchoXFlow, which remain governed by their own terms.
