"""
predict.py — Asthma Poor-Control Prediction
============================================
Command-line script to generate poor-control predictions on new patient data.

Usage
-----
    python predict.py --input new_patients.csv --output predictions.csv

Arguments
---------
    --input   Path to the input CSV (processed, output of 01_EDA.ipynb)
    --output  Path to write the predictions CSV (default: predictions.csv)
    --models  Path to the models/ directory (default: models/)
    --proba   If set, include calibrated probabilities in the output

Output columns
--------------
    poor_control_pred   Binary prediction (0 = good control, 1 = poor control)
    poor_control_proba  Calibrated probability of poor control (only with --proba)

Example
-------
    python predict.py --input data/cohort_2025.csv --output results/predictions_2025.csv --proba
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate asthma poor-control predictions on new patient data.'
    )
    parser.add_argument('--input',  required=True,          help='Path to input CSV')
    parser.add_argument('--output', default='predictions.csv', help='Path to output CSV')
    parser.add_argument('--models', default='models/',      help='Path to models directory')
    parser.add_argument('--proba',  action='store_true',    help='Include calibrated probabilities in output')
    return parser.parse_args()


# ── Load artefacts ────────────────────────────────────────────────────────────

def load_artefacts(models_dir: str) -> dict:
    """
    Load all model artefacts from the models/ directory.

    Expected files:
        model_tuned.pkl   — tuned XGBoost classifier
        iso_reg.pkl       — isotonic regression calibrator
        encoders.pkl      — target encoders, OHE, feature metadata
        threshold.pkl     — optimal decision threshold (recall >= 0.80)
    """
    required = ['model_tuned.pkl', 'iso_reg.pkl', 'encoders.pkl', 'threshold.pkl']
    for fname in required:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            sys.exit(f'[ERROR] Artefact not found: {path}\n'
                     f'Run 02_modeling.ipynb first to generate the models/ directory.')

    model     = joblib.load(os.path.join(models_dir, 'model_tuned.pkl'))
    iso_reg   = joblib.load(os.path.join(models_dir, 'iso_reg.pkl'))
    encoders  = joblib.load(os.path.join(models_dir, 'encoders.pkl'))
    threshold = joblib.load(os.path.join(models_dir, 'threshold.pkl'))

    return {
        'model':            model,
        'iso_reg':          iso_reg,
        'target_encoders':  encoders['target_encoders'],
        'ohe':              encoders['ohe'],
        'global_mean':      encoders['global_mean'],
        'ohe_cols':         encoders['ohe_cols'],
        'drop_features':    encoders['drop_features'],
        'features':         encoders['features'],
        'threshold':        threshold,
    }


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, artefacts: dict) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a raw processed DataFrame.

    Steps:
        1. Drop non-feature columns (target, year — not available at inference time)
        2. Target encoding for area and ap_zone
        3. One-hot encoding for eos_level, ige_level and their lags
        4. Drop zero-importance features (identified via SHAP analysis)
        5. Reorder columns to match the training feature order

    Args:
        df:         DataFrame with processed patient-year records (output of EDA)
        artefacts:  Dict returned by load_artefacts()

    Returns:
        Feature matrix ready for model.predict_proba()
    """
    X = df.drop(columns=['poor_control', 'year'], errors='ignore').copy()

    # Target encoding — unseen categories receive the global mean
    for col, mapping in artefacts['target_encoders'].items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(artefacts['global_mean'])

    # One-hot encoding
    ohe_cols = artefacts['ohe_cols']
    ohe_cols_present = [c for c in ohe_cols if c in X.columns]
    if ohe_cols_present:
        encoded    = artefacts['ohe'].transform(X[ohe_cols_present])
        encoded_df = pd.DataFrame(
            encoded,
            columns=artefacts['ohe'].get_feature_names_out(ohe_cols_present),
            index=X.index
        )
        X = pd.concat([X.drop(columns=ohe_cols_present), encoded_df], axis=1)

    # Drop zero-importance features
    X = X.drop(columns=[c for c in artefacts['drop_features'] if c in X.columns])

    # Align to training feature order
    missing_cols = set(artefacts['features']) - set(X.columns)
    if missing_cols:
        sys.exit(f'[ERROR] Missing columns in input data: {sorted(missing_cols)}\n'
                 f'Make sure the input CSV is the output of 01_EDA.ipynb.')

    return X[artefacts['features']]


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, artefacts: dict, return_proba: bool = False):
    """
    Generate poor-control predictions for a DataFrame of patient-year records.

    Pipeline: preprocess → XGBoost → isotonic calibration → threshold

    Args:
        df:           Processed DataFrame (output of EDA pipeline)
        artefacts:    Dict returned by load_artefacts()
        return_proba: If True, also return calibrated probability scores

    Returns:
        preds:  Series[int]   — 0 = good control, 1 = poor control
        probas: Series[float] — calibrated probability (only if return_proba=True)
    """
    X         = preprocess(df, artefacts)
    proba_raw = artefacts['model'].predict_proba(X)[:, 1]
    proba_cal = artefacts['iso_reg'].predict(proba_raw)
    preds     = (proba_cal >= artefacts['threshold']).astype(int)

    preds_s = pd.Series(preds,     index=df.index, name='poor_control_pred')
    proba_s = pd.Series(proba_cal, index=df.index, name='poor_control_proba')

    return (preds_s, proba_s) if return_proba else preds_s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load input data
    if not os.path.exists(args.input):
        sys.exit(f'[ERROR] Input file not found: {args.input}')

    print(f'[INFO] Loading input data: {args.input}')
    df = pd.read_csv(args.input)
    print(f'[INFO] Records to score: {len(df):,}')

    # Load artefacts
    print(f'[INFO] Loading model artefacts from: {args.models}')
    artefacts = load_artefacts(args.models)
    print(f'[INFO] Decision threshold: {artefacts["threshold"]:.3f}')

    # Generate predictions
    print('[INFO] Generating predictions...')
    if args.proba:
        preds, probas = predict(df, artefacts, return_proba=True)
        results = pd.DataFrame({
            'poor_control_pred':  preds.values,
            'poor_control_proba': probas.values.round(4),
        }, index=df.index)
    else:
        preds = predict(df, artefacts, return_proba=False)
        results = pd.DataFrame({'poor_control_pred': preds.values}, index=df.index)

    # Summary
    n_flagged = (results['poor_control_pred'] == 1).sum()
    print(f'[INFO] Patients flagged as poor control: {n_flagged:,} / {len(df):,} '
          f'({n_flagged / len(df):.1%})')

    # Save output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f'[INFO] Predictions saved to: {args.output}')


if __name__ == '__main__':
    main()
