# Doppler-free triage for elevated left atrial pressure

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22994944.svg)](https://doi.org/10.5281/zenodo.22994944)

Code for estimating **elevated left atrial pressure from B-mode echocardiographic video alone — no Doppler
at inference** — developed on the public MIMIC-IV-ECHO dataset and validated, without adaptation, on the
independent public EchoXFlow dataset (Akershus University Hospital, Norway).

This repository accompanies the manuscript *"Doppler-Free Triage for Elevated Left Atrial Pressure from
B-Mode Echocardiographic Video: Development on Open Data with Independent External Validation on Raw
Beamspace Recordings"* (Köksal F., submitted). An earlier four-class grading version of this work is preserved in
the git history (commit `816a474`).

| | Internal test (MIMIC-IV-ECHO) | External (EchoXFlow) |
|---|---|---|
| Guideline-defined elevated LAP | AUROC 0.882 (0.839–0.920), n = 438 | AUROC 0.924 (0.867–0.968), n = 191 |
| Elevated E/e′ (prespecified primary, external) | AUROC 0.824 (0.774–0.868) | AUROC 0.829 (0.775–0.878), n = 303 |

All reported numbers are produced by `external/canonical_results.py` (fixed seed, 4,000 bootstrap
resamples) and stored in [`external/results/`](external/results/). Prespecified secondary analyses
(plan §5.6, run in v2.1.0): cross-fitted recalibration in the external cohort restores calibration without
changing discrimination; histogram matching of image intensity raises the external AUROC to 0.854 (elevated
E/e′) and 0.949 (guideline LAP). The zero-shot results above remain primary; see
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

---

## Prespecification and deviations

- [`docs/analysis_plan.md`](docs/analysis_plan.md) — written before any external data were analysed:
  endpoints, reference standards, cohort rules, decision parameters, the quality covariate and its
  tertile boundaries.
- [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) — the dated record of every departure from that plan.
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

Scripts run from the project root. Paths default to `~/mimic-echo/`, `~/echoxflow-render/` and
`echoxflow/`; machine-specific roots can be set with `MIMIC_IV_ROOT`, `MIMIC_ECHO_DICOM_ROOT` and
`GCP_BILLING_PROJECT`.

## Requirements

Python 3.13, PyTorch 2.11 (CUDA 12.8), scikit-learn, pydicom, OpenCV, zarr, numcodecs, matplotlib; see
`requirements.txt`. A single 24 GB GPU is sufficient. The video encoder is **PanEcho** (Holste et al.,
*JAMA* 2025), loaded from `torch.hub` (`CarDS-Yale/PanEcho`); the view classifier is from **EchoPrime**
(Vukadinovic et al., *Nature* 2025). Neither is vendored here. EchoXFlow reading and scan conversion use
the dataset's reference implementation (<https://github.com/Ahus-AIM/EchoXFlow>).

## Use of a large language model

A large language model (Claude, Anthropic) operated as a coding and analysis agent and wrote and executed
much of this code under the author's direction. Its role and the human quality control applied are
described in the Methods of the manuscript.

## Citation

All archived versions: https://doi.org/10.5281/zenodo.22994944 (resolves to the latest release; v2.0.0: https://doi.org/10.5281/zenodo.22994945).
See `CITATION.cff`. Please also cite MIMIC-IV-ECHO, MIMIC-IV, PhysioNet and EchoXFlow.

## Licence

Code: MIT (see `LICENSE`). The licence covers **this code only** and confers no rights over MIMIC-IV-ECHO,
MIMIC-IV or EchoXFlow, which remain governed by their own terms.
