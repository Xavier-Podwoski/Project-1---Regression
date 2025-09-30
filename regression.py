#!/usr/bin/env python3
"""
run_regression.py

Downloads the UCI Energy Efficiency dataset, performs preprocessing, splits into
train/val/test (7:1:2), trains Linear Regression and Ridge Regression models
(using scikit-learn), evaluates them (MSE, MAE, R2), and saves results and models.

Usage:
    python regression.py         # runs full pipeline
    python regression.py --no-download  # if dataset is already present as ENB2012_data.xlsx in data/

Outputs (in ./outputs):
    - metrics.json       : evaluation numbers
    - models/            : saved sklearn models (joblib)
    - results.csv        : predictions + true values for test set
    - report.pdf         : filled report (replaces placeholders)
"""
import argparse
import os
from pathlib import Path
import zipfile
import io
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/242/energy%2Befficiency.zip"
EXCEL_NAME = "ENB2012_data.xlsx"

def download_and_extract():
    print("Downloading dataset from UCI...")
    r = requests.get(UCI_ZIP_URL, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if name.endswith(".xlsx") or name.endswith(".xls"):
            print("Extracting", name)
            z.extract(name, path=DATA_DIR)
            extracted = DATA_DIR / name
            if extracted.exists() and extracted.parent != DATA_DIR:
                extracted.rename(DATA_DIR / Path(name).name)
            return DATA_DIR / Path(name).name
    raise FileNotFoundError("Excel file not found inside zip")

def load_data(excel_path):
    print("Loading data from", excel_path)
    df = pd.read_excel(excel_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 10:
        raise ValueError("Unexpected format: numeric cols found: " + str(numeric_cols))
    df = df[numeric_cols[:10]]
    df.columns = ['X1','X2','X3','X4','X5','X6','X7','X8','Y1','Y2']
    return df

def split_data(df, random_state=42):
    X = df[['X1','X2','X3','X4','X5','X6','X7','X8']].values
    y = df['Y1'].values  # predicting Heating Load
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    relative_val = 0.1 / (0.1 + 0.7)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val, random_state=random_state)
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    results = {}
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge_alpha1.0': Ridge(alpha=1.0)
    }
    for name, model in models.items():
        print("Training", name)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        metrics = {}
        for split_name, y_true, y_pred in [('val', y_val, y_val_pred), ('test', y_test, y_test_pred)]:
            metrics[split_name] = {
                'MSE': float(mean_squared_error(y_true, y_pred)),
                'MAE': float(mean_absolute_error(y_true, y_pred)),
                'R2' : float(r2_score(y_true, y_pred))
            }
        results[name] = metrics
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    best_model_name = min(results.keys(), key=lambda n: results[n]['val']['MSE'])
    best_model = joblib.load(MODELS_DIR / f"{best_model_name}.joblib")
    y_test_pred_best = best_model.predict(X_test)
    out_df = pd.DataFrame(X_test, columns=['X1','X2','X3','X4','X5','X6','X7','X8'])
    out_df['Y_true'] = y_test
    out_df['Y_pred'] = y_test_pred_best
    out_df.to_csv(OUT_DIR / "results_test.csv", index=False)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({'generated_at': datetime.utcnow().isoformat(), 'results': results, 'best_model': best_model_name}, f, indent=2)
    return results

def create_report(results, out_path):
    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Energy Efficiency Regression Report")
    c.setFont("Helvetica", 10)
    y -= 30
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Problem")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Predict heating load (Y1) from 8 building features (X1..X8).")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Data")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Dataset: UCI Energy Efficiency. 768 instances, 8 features, 2 targets (heating/cooling loads).")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Method")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Split: 70% train / 10% val / 20% test. Models: Linear Regression & Ridge Regression (alpha=1.0). Metrics: MSE, MAE, R2.")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Results")
    y -= 16
    c.setFont("Helvetica", 10)
    rtext = json.dumps(results, indent=2)
    for line in rtext.splitlines():
        if y < margin + 40:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:110])
        y -= 12
    c.showPage()
    c.save()

def main(no_download=False):
    excel_path = DATA_DIR / EXCEL_NAME
    if not no_download:
        try:
            excel_path = download_and_extract()
        except Exception as e:
            print("Download failed:", e)
            if excel_path.exists():
                print("Proceeding with existing file", excel_path)
            else:
                raise
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {excel_path}. Please place it into {DATA_DIR}")
    df = load_data(excel_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    create_report(results, OUT_DIR / "report.pdf")
    print("Done. Outputs are in", OUT_DIR)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-download", action="store_true", help="Use local data without downloading")
    args = p.parse_args()
    main(no_download=args.no_download)
