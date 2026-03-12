# Survival Analysis of Biological Treatment in Severe Asthma

Time-to-event analysis of key clinical outcomes in severe asthma patients
candidates for biological treatment, based on a prospective real-world cohort
from Cantabria, Spain.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![lifelines](https://img.shields.io/badge/lifelines-0.30.0-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)


## Overview
Asthma is a chronic respiratory disease affecting tens of thousands of patients 
in the Cantabria region of Spain. Despite being a manageable condition, a 
significant proportion of patients experience poor clinical control each year, 
leading to exacerbations, emergency visits and hospitalisations that are largely 
preventable with timely intervention.

This project applies **time-to-event (survival) analysis** to characterise the clinical
trajectory of severe asthma patients candidates for biological treatment, estimating
when key events occur and which baseline factors are associated with them.

The dataset covers severe asthma patients in Cantabria candidates for biological treatment
(Benralizumab or Tezepelumab), prospectively enrolled at the asthma unit with active and
ongoing inclusion. Key variables include baseline clinical phenotype, pulmonary
function, inflammatory biomarkers, comorbidities, and biological treatment records.


## Pipeline

The project follows a sequential pipeline from raw EHR data to production-ready predictions:

| Step | Notebook | Key decision |
|------|----------|-------------|
| **Preprocessing** | `00_preprocessing.ipynb` | Longitudinal-to-baseline collapse per patient, survival time variables built from raw visit records, binary/categorical encoding and median imputation |
| **Time to Biological** | `01_time_to_biological.ipynb` | T0 = first asthma unit visit, event = first biological initiation, censored if no biological received |
| **Time to Exacerbation** | `02_time_to_exacerbation.ipynb` | T0 = biological initiation, event = first visit with severe exacerbation after T0, extracted by iterating longitudinal visit records |
| **Time to Hospitalisation** | `02_time_to_exacerbation.ipynb` | T0 = biological initiation, event = first visit with asthma-related hospitalisation after T0, secondary endpoint within the same notebook |
| **Time to Biological Failure** | `03_time_to_biological_failure.ipynb` | T0 = biological initiation, event = biological withdrawal (`FechaRetirada_bio`), censored if still on treatment at last visit |


## Repository Structure
```
Asthma-Control-ML/
├── Poor_Control_Prediction/
│   ├── models/
│       ├── encoders.pkl          # Target encoders, OHE and feature metadata
│       ├── iso_reg.pkl           # Isotonic regression calibrator
│       ├── model_calibrated.pkl  # XGBoost + calibration wrapper
│       ├── model_tuned.pkl       # Tuned XGBoost classifier
│       └── threshold.pkl         # Optimal decision threshold (recall ≥ 0.80)
│
│   ├── notebooks/
│       ├── 01_EDA.ipynb          # Exploratory analysis, preprocessing and feature engineering
│       ├── 02_modeling.ipynb     # Training, SHAP analysis, tuning and calibration
│       └── 03_inference.ipynb    # Inference pipeline demonstration
│
│   ├── src/               
│       └── predict.py     # Production-ready prediction script (CLI)
│   └── README.md
│
├── data/
│   └── .gitkeep              # Data folder versioned without data (clinical records, not uploaded)
├── requirements.txt          # Python dependencies
├── environment.yml           # Python dependencies for conda environment
└── .gitignore
```


## Quickstart

### Installation

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate clinical-ml
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
python Poor_Control_Prediction/src/predict.py \
    --input data/patients.csv \
    --output predictions.csv \
    --models Poor_Control_Prediction/models/ \
    --proba
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
