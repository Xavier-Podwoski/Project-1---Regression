#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_energy_data(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please download it from "
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx "
            "and place it in the 'data/' folder."
        )
    df = pd.read_excel(data_path, engine="openpyxl")
    X = df.iloc[:, 0:8]
    Y = df.iloc[:, 8:10]
    return X, Y

def split_data(X: pd.DataFrame, y: pd.DataFrame, seed: int = 42):
    # First, split off test set (20% of full data)
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    # X_rem is 80%; from that we take 1/8 (~0.125 of total) as validation,
    # which is (0.125 / 0.80) = 0.15625 of X_rem
    val_fraction = 0.125 / 0.80  # = 0.15625
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, test_size=val_fraction, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_metrics(y_true, y_pred, prefix: str = ""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix}MSE: {mse:.4f}")
    print(f"{prefix}RMSE: {rmse:.4f}")
    print(f"{prefix}MAE: {mae:.4f}")
    print(f"{prefix}R^2: {r2:.4f}")

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, alpha: float = 1.0):
    # Train linear regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    
    # Train ridge regression
    rid = Ridge(alpha=alpha)
    rid.fit(X_train, y_train)
    
    print("== On Validation Set ==")
    for name, model in [("Linear", lin), ("Ridge", rid)]:
        print(f"--- {name} Regression ---")
        yv_pred = model.predict(X_val)
        print_metrics(y_val, yv_pred, prefix="  ")
    
    print("\n== On Test Set ==")
    for name, model in [("Linear", lin), ("Ridge", rid)]:
        print(f"--- {name} Regression ---")
        yt_pred = model.predict(X_test)
        print_metrics(y_test, yt_pred, prefix="  ")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, os.pardir, "data", "ENB2012_data.xlsx")
    
    X, y = load_energy_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("Data split sizes:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Val:  ", X_val.shape, y_val.shape)
    print("  Test: ", X_test.shape, y_test.shape)
    
    train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, alpha=1.0)

if __name__ == "__main__":
    main()
