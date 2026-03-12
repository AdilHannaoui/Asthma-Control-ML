# Asthma-Clinical-ML

Clinical machine learning and statistical analysis applied to severe asthma,
based on real-world Electronic Health Record (EHR) data from the Cantabria
region of Spain.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains two independent clinical studies on severe asthma,
addressing complementary questions across different patient cohorts and
methodological approaches:

- **Poor Control Prediction** applies supervised machine learning to anticipate
  which patients will experience poor asthma control in the following year,
  enabling proactive clinical prioritisation.

- **Survival Analysis** applies time-to-event statistical methods to characterise
  the clinical trajectory of severe asthma patients on biological treatment,
  estimating when key outcomes occur and which baseline factors are associated
  with them.

Together, they cover two distinct clinical problems: early identification of
at-risk patients, and longitudinal characterisation of treatment outcomes.

## Projects

### Poor Control Prediction
> Can we predict which patients will lose asthma control next year?

End-to-end ML pipeline (XGBoost) trained on 304,483 patient-year records
(2018–2024). Optimised for recall ≥ 0.80 to minimise missed deteriorations.
Includes probability calibration for direct clinical interpretability.

| | |
|--|--|
| **Cohort** | Asthmatic patients in Cantabria, 2018–2024 |
| **Method** | XGBoost + Isotonic calibration + SHAP |
| **Key result** | ROC-AUC 0.82 · Recall 0.83 on held-out 2024 test set |
| **README** | [Poor_Control_Prediction/README.md](Poor_Control_Prediction/README.md) |

---

### Survival Analysis of Biological Treatment
> How long until a patient exacerbates, is hospitalised, or fails their biological?

Time-to-event analysis (Kaplan-Meier + Cox PH) on a prospective cohort of
severe asthma patients candidates for biological treatment (Benralizumab /
Tezepelumab) in Cantabria. Three endpoints analysed from biological initiation.

| | |
|--|--|
| **Cohort** | Severe asthma patients candidates for biological treatment, Cantabria |
| **Method** | Kaplan-Meier · Log-rank tests · Cox Proportional Hazards |
| **Key result** | Median 9.4 months to biological · 14.5 months to first exacerbation · 15.0 months to failure |
| **README** | [Survival_Analysis/README.md](Survival_Analysis/README.md) |

## Repository Structure
```
Asthma-Clinical-ML/
├── Poor_Control_Prediction/
│   ├── models/                        # Serialised model artefacts
│   ├── notebooks/                     # EDA, modelling and inference
│   ├── src/predict.py                 # Production CLI prediction script
│   └── README.md
├── Survival_Analysis/
│   ├── notebooks/                     # Preprocessing and three analyses
│   ├── plots/                         # Kaplan-Meier curves and forest plots
│   └── README.md
├── data/
│   └── .gitkeep                       # Data folder (clinical records not uploaded)
├── Dockerfile                         # Container for poor control prediction
├── environment.yml                    # Conda environment
├── requirements.txt                   # pip dependencies
└── .gitignore
```

## Installation

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate clinical-ml
```

Or using pip:
```bash
pip install -r requirements.txt
```

For the poor control prediction pipeline, a Docker image is also available:
```bash
docker build -t asthma-control-ml .
```

## Data

Clinical records are not included in this repository. The `data/` folder is
versioned empty via `.gitkeep`. Both projects expect their respective raw data
files to be placed locally before running the notebooks — see each project's
README for details.

## License

MIT License — see [LICENSE](LICENSE) for details.
