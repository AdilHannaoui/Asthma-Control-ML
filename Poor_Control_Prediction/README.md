# Clinical Control Prediction in Asthmatic Patients

End-to-end ML pipeline for the annual prediction of poor asthma control
based on real-world Electronic Health Record (EHR) data from a 7-year cohort.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)


## Overview
Asthma is a chronic respiratory disease affecting tens of thousands of patients 
in the Cantabria region of Spain. Despite being a manageable condition, a 
significant proportion of patients experience poor clinical control each year, 
leading to exacerbations, emergency visits and hospitalisations that are largely 
preventable with timely intervention.

This project builds a supervised ML model to **predict which patients will 
experience poor asthma control in the following year**, enabling clinicians to 
proactively prioritise follow-up before deterioration occurs.

The dataset covers asthmatic patients in Cantabria between 2018 and 2024 
(304,483 patient-year records). Key variables include pneumology and allergy 
follow-up, chronic comorbidities, treatment regimens, and biological markers 
such as eosinophil levels and IgE.


## Pipeline

The project follows a sequential pipeline from raw EHR data to production-ready predictions:

| Step | Notebook | Key decision |
|------|----------|-------------|
| **EDA & Preprocessing** | `01_EDA.ipynb` | Patient ID anonymisation, eosinophil/IgE consolidation into ordinal variables, temporal lag features with gap-aware handling |
| **Modelling** | `02_modeling.ipynb` | Temporal train/val/test split (2018–2022 / 2023 / 2024), target encoding, XGBoost baseline |
| **SHAP Analysis** | `02_modeling.ipynb` | Zero-importance feature removal, clinical interpretability of top predictors |
| **Hyperparameter Tuning** | `02_modeling.ipynb` | Optuna with 150 + 100 zoom-in trials, GPU acceleration |
| **Probability Calibration** | `02_modeling.ipynb` | Isotonic regression to correct score inflation from `scale_pos_weight` |
| **Inference** | `03_inference.ipynb` / `predict.py` | Full pipeline encapsulated for production use |


## Repository Structure
```
Asthma-Control-ML/
│
├── data/
│   └── .gitkeep              # Data folder versioned without data (clinical records, not uploaded)
│
├── models/
│   ├── encoders.pkl          # Target encoders, OHE and feature metadata
│   ├── iso_reg.pkl           # Isotonic regression calibrator
│   ├── model_calibrated.pkl  # XGBoost + calibration wrapper
│   ├── model_tuned.pkl       # Tuned XGBoost classifier
│   └── threshold.pkl         # Optimal decision threshold (recall ≥ 0.80)
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory analysis, preprocessing and feature engineering
│   ├── 02_modeling.ipynb     # Training, SHAP analysis, tuning and calibration
│   └── 03_inference.ipynb    # Inference pipeline demonstration
│
├── predict.py                # Production-ready prediction script (CLI)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```


## Quickstart

### Installation

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate ml-clinico
```

Or using pip:
```bash
pip install -r requirements.txt
```

Or using Docker:
```bash
docker build -t asthma-control-ml .
```


### Generate predictions

**Standard:**
```bash
python predict.py --input data/patients.csv --output predictions.csv --proba
```

**Docker:**
```bash
docker run -v $(pwd)/data:/app/data asthma-control-ml \
    --input data/patients.csv \
    --output data/predictions.csv \
    --proba
```
### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ✓ | — | Path to input CSV (processed, output of `01_EDA.ipynb`) |
| `--output` | ✗ | `predictions.csv` | Path to write predictions |
| `--models` | ✗ | `models/` | Path to models directory |
| `--proba` | ✗ | `False` | Include calibrated probabilities in output |

### Output

| Column | Description |
|--------|-------------|
| `poor_control_pred` | Binary prediction (0 = good control, 1 = poor control) |
| `poor_control_proba` | Calibrated probability of poor control (only with `--proba`) |


## Clinical Context

### Why predict poor control?

Poor asthma control is defined as the occurrence of any of the following in a given year:
- Oral corticosteroid (OCS) use
- Magnesium sulphate use
- ICU admission or asthma-related hospitalisation
- AMR < 0.5 and/or SABA ≥ 12 dispensed inhalers

In the Cantabria cohort, **15.65% of patient-years meet this definition**, with a growing trend from 13.4% (2018) to 18.4% (2023–2024). Identifying these patients in advance allows clinicians to intervene before deterioration occurs.

### Why recall over precision?

The model is optimised for **recall ≥ 0.80** on the positive class rather than overall accuracy, reflecting the asymmetric cost of each error type:

| Error type | Clinical consequence | Cost |
|------------|---------------------|------|
| False negative | Poor control goes undetected → A&E visit, hospitalisation | **High** |
| False positive | Unnecessary follow-up visit | Low |

At the chosen threshold the model detects ~80% of poor-control patients, at the cost of ~35% precision among those flagged — an acceptable trade-off for an early-warning screening tool where the cost of missing a case far outweighs the cost of an extra visit.

### Probability calibration

XGBoost with `scale_pos_weight` tends to inflate raw probability scores. **Isotonic regression calibration** is applied to ensure that predicted probabilities are directly interpretable: a score of 0.30 means ~30% real risk of poor control. This is essential for clinical use, where clinicians interpret probability scores literally.


## Results

All metrics are reported on the **held-out test set (2024)**, which was not used at any point during model development or tuning.

### Baseline vs Tuned

| Metric | Baseline | Tuned |
|--------|----------|-------|
| ROC-AUC (val) | 0.8233 | 0.8241 |
| PR-AUC (val) | 0.5551 | 0.5583 |

### Validation vs Test

| Metric | Validation (2023) | Test (2024) |
|--------|------------------|-------------|
| ROC-AUC | 0.8241 | 0.8195 |
| PR-AUC | 0.5583 | 0.5344 |

The minimal drop between validation and test (~0.005 ROC-AUC) indicates that the model generalises well across years despite the temporal drift in poor-control rates observed in the dataset.

### Classification report — Test set (threshold = 0.139, calibrated)

| | Precision | Recall | F1 |
|--|-----------|--------|-----|
| Good control | 0.94 | 0.65 | 0.77 |
| Poor control | 0.34 | 0.83 | 0.48 |
