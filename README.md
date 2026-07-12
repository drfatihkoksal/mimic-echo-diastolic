# Doppler-free diastolic grading on MIMIC-IV-ECHO

Code for grading left ventricular diastolic function from **B-mode echocardiographic video alone
— no Doppler** — on the public MIMIC-IV-ECHO dataset, with reference grades derived from the
**2025 ASE guideline**.

This repository accompanies the manuscript *"Automated Grading of Left Ventricular Diastolic
Function from B-Mode Echocardiographic Video: A Reproducible Open Benchmark on MIMIC-IV-ECHO
Aligned with the 2025 ASE Guideline"* (Köksal F., under review).

---

## ⚠️ No data in this repository — and why

MIMIC-IV-ECHO and MIMIC-IV are **credentialed-access** datasets distributed by PhysioNet under a
data use agreement that prohibits redistributing the data or patient-level data derived from it.

This repository therefore contains **code only**. It contains no images, no labels, no splits, no
cohort tables and no model weights. What it contains instead is the code that **regenerates all of
them deterministically** from the source data, so that any PhysioNet-credentialed user can
reproduce every number in the paper without anyone having to redistribute a single patient record.

To run this you need your own credentialed access:

1. Complete the CITI "Data or Specimens Only Research" training.
2. Obtain credentialed access to **MIMIC-IV-ECHO v1.0** (https://doi.org/10.13026/nrjh-5r77),
   **MIMIC-IV** (https://doi.org/10.13026/6mm1-ek67) and **MIMIC-IV-ECG**, and sign their DUAs.
3. Download them yourself. Nothing here will download restricted data for you.

The reference labels, the patient-level splits and the trained weights are *derived* artefacts of
those datasets. If you want them as files rather than regenerating them, they must be obtained
through PhysioNet's own channel for derived data, not from this repository.

---

## What the pipeline does

| Stage | Script | Purpose |
|---|---|---|
| Labels | `diastolic_grading.py` | Applies the 2025 ASE algorithm to the structured measurements; excludes atrial fibrillation (via temporally matched MIMIC-IV-ECG, ICD fallback) and significant mitral disease, for which the algorithm is undefined. Returns `Incomputable` when a determinant is missing — it never imputes a grade. |
| Preprocess | `shrink_pipeline.py`, `run_local.py`, `redownload_hires.py` | DICOM → grayscale B-mode cine → sector crop → 224×224 NPZ. Color/spectral Doppler and still frames are dropped from the DICOM metadata, never from pixel analysis. |
| Views | `view_classify.py` | Keeps A4C / A2C / PLAX only. |
| Splits | `make_splits.py` | Patient-level, grade-stratified 70/15/15 with zero patient overlap. |
| Model | `model.py`, `dataset.py`, `train.py` | PanEcho video encoder fine-tuned with a CORN ordinal head plus auxiliary regression of the guideline determinants (**training signal only** — no measurement and no Doppler are needed at inference). |
| Fusion | `embed_clips.py`, `train_mil.py`, `mil_predict.py` | Gated-attention multiple-instance pooling over a study's clips → the reported model. |
| Evaluation | `binary_eval.py`, `compare_models.py`, `panecho_baseline.py`, `ablation_ef.py` | Screening AUROCs with bootstrap CIs, paired model comparison, zero-shot and EF-only baselines. |
| Out-of-fold | `kfold_oof.py`, `run_oof_resume.sh` | 5-fold patient-grouped cross-validation giving an unbiased prediction for every study — the basis of the prognostic analyses. |
| Prognosis | `mortality_analysis.py`, `ntprobnp_analysis.py`, `run_analysis_chain.sh` | One-year all-cause mortality (Cox, C-index) and NT-proBNP — two references *outside* the echocardiogram. |
| Ensemble | `ensemble_eval.py`, `run_ensemble_chain.sh` | Five models on independent partitions; robustness and soft-voting. |
| Figures | `make_figures.py` | Every figure in the paper, regenerated from the data at draw time. |

### Order

```bash
python3 diastolic_grading.py            # labels  (needs MIMIC-IV hosp/ + MIMIC-IV-ECG)
python3 shrink_pipeline.py              # DICOM -> NPZ
python3 view_classify.py                # A4C/A2C/PLAX
python3 make_splits.py                  # patient-level splits
python3 train.py                        # clip-level encoder
python3 embed_clips.py && python3 train_mil.py   # attention-MIL fusion (the reported model)
python3 binary_eval.py                  # test-set metrics
python3 kfold_oof.py --folds 5          # out-of-fold predictions (~4 h on one GPU)
python3 mortality_analysis.py --preds ~/mimic-echo/runs/oof_predictions.csv
python3 ntprobnp_analysis.py --preds ~/mimic-echo/runs/oof_predictions.csv
python3 make_figures.py fig1            # ... fig2, fig3, fig4, figS1, figS2, central
```

Paths default to `~/mimic-echo/`; override them with the scripts' own flags.

### Long GPU jobs

`run_oof_resume.sh`, `run_ensemble_chain.sh` and `run_analysis_chain.sh` wrap the long jobs.
They checkpoint **every epoch** (atomically) and wait for the GPU to come back before spending a
retry — written after two multi-hour runs were lost to a GPU that dropped off the PCIe bus under
sustained load. If your hardware is stable you will never notice them; if it is not, they are the
difference between finishing and not.

## Requirements

Python 3.13, PyTorch 2.11 (CUDA 12.8), scikit-learn, lifelines, pydicom, OpenCV, matplotlib.
See `requirements.txt`. A single 24 GB+ GPU is enough; the MIL head trains on CPU in minutes.

The video encoder is **PanEcho** (Holste et al., *JAMA* 2025), pulled from `torch.hub`
(`CarDS-Yale/PanEcho`) — it is not vendored here.

## Citation

If you use this code, please cite the manuscript (details will be added on acceptance) and, per the
PhysioNet terms, the datasets themselves and the PhysioNet paper.

## Licence

Code: MIT (see `LICENSE`). The licence covers **this code only**. It confers no rights whatsoever
over MIMIC-IV-ECHO or MIMIC-IV, which remain governed by their own data use agreements.
