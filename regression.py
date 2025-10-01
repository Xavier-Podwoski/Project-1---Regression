#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd

class LinearRegression: # initalize the model for linear regression
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, x, y):
        # Add Bias
        x_bias = np.column_stack([np.ones(x.shape[0]), x])

        # Optimization formula for Linear Regression: (X^T X)^(-1) X^T y
        xtx = x_bias.T @ x_bias
        xty = x_bias.T @ y
        w_all = np.linalg.solve(xtx, xty)

        # Separate bias and weights from each other
        self.bias = w_all[0]
        self.weight = w_all[1:]
        return self

    #Intial predictions
    def prediction(self, x):
        return x @ self.weight + self.bias

class RidgeRegression: # Initalize the model for ridge regression
    def __init__(self, strength = 1.0): # Using 1.0 for simplicity's sake
        self.strength = strength
        self.weight = None
        self.bias = None
    
    def fit(self, x, y):
        n_sample, n_features = x.shape
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        total_f = x_bias.shape[1]
        reg_matrix = np.eye(total_f) * self.strength
        reg_matrix[0, 0] = 0


        # (X^T*X + lamda(I))^-1 * X^T * Y Ridge Regression formula optimized
        # Variables used are from function and not from formula given in notes, but yield same purpose
        n = x_bias.T @ x_bias + reg_matrix
        m = x_bias.T @ y
        w_all = np.linalg.solve(n, m)
        self.bias = w_all[0]
        self.weight = w_all[1:]
        return self
    
    # Makes new predictions based on new data 
    def prediction(self, x):
        return x @ self.weight + self.bias
      
    # Test the model
    def evaluate(self, a, p):
        t = a - p
        mean_e = np.mean(t**2)
        root_mean_e = np.sqrt(mean_e)
        mean_absolute_e = np.mean(np.abs(t))
        sum_residuals = np.sum(t**2)
        sum_total = np.sum((a - np.mean(a))**2)
        r2 = 1 - (sum_residuals / sum_total)

        # Returns given metrics that were solved
        return {
            'Mean squared error': mean_e,
            'Root mean squared error': root_mean_e,
            'Mean absolute error': mean_absolute_e,
            'R2 score': r2
        }

# Split the data
def data_split(x, y, t_r = 0.7, v_r = 0.1, te_r = 0.2, t_s = 7):
    np.random.seed(t_s) 
    n_sample = x.shape[0]
    index = np.random.permutation(n_sample)
    n_train = int(n_sample * t_r)
    n_val = int(n_sample * v_r)
    
    train_index = index[:n_train]
    val_index = index[n_train:n_train+n_val]
    test_index = index[n_train+n_val:]
    x_train, y_train = x[train_index], y[train_index]
    x_val, y_val = x[val_index], y[val_index]    
    x_test, y_test = x[test_index], y[test_index]

    return x_train, x_val, x_test, y_train, y_val, y_test

#Data Normalization
def data_normalization(x_train, x_val, x_test):
    #calc mean and std
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    #handle 
    std = np.where(std == 0, 1, std)
    #(x - mean) / std
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_val, x_test

#evaluation function
def evaluate_model(y_true, y_pred):
    t = y_true - y_pred
    mean_e = np.mean(t**2)
    root_mean_e = np.sqrt(mean_e)
    mean_absolute_e = np.mean(np.abs(t))
    sum_residuals = np.sum(t**2)
    sum_total = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (sum_residuals / sum_total)
    
    return {
        'Mean Squared Error': mean_e,
        'Root Mean Squared Error': root_mean_e,
        'Mean Absolute Error': mean_absolute_e,
        'R-squared': r2,
    }

#solve ridge , see code from book
def solve_ridge(x, y, lamdaMaxValue = 4500):
    #lists for ridge and df
    ridge_list = []
    df_list = []
    #transpose
    xtx = x.T @ x
    xty = x.T @ y
    
    #single val 
    _, svd, _ = np.linalg.svd(x, full_matrices=False)
    svd_sq = svd**2

    #loop
    for i in range(lamdaMaxValue + 1):
        #coefficient = (X^T*X + lamda(I))^-1 * X^T * Y
        reg_mat = np.identity(xtx.shape[0]) * i
        coeff = np.linalg.solve(xtx + reg_mat, xty)
        ridge_list.append(coeff)
        df = np.sum(svd_sq / (svd_sq + i))
        df_list.append(df)
        
    return ridge_list, df_list

#main function
def data_load():
    # load data 
    try:
        #import 
        df = pd.read_excel('ENB2012_data.xlsx')
        print(f"Dataset successfully loaded: {df.shape}")
        
        # extract data 
        f_colum = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        x = df[f_colum].values
        y1 = df['Y1'].values
        y2 = df['Y2'].values
        
        print(f"Features: {f_colum}")
        print(f"Y1 represents heating load, Y2 represents cooling load")
        
        return x, y1, y2
        
    except FileNotFoundError:
        print("The current file you are looking for does not exist here.cPlease download the file locally and store in data folder")
        return None, None, None

def main():
    # load data
    x, y1, y2 = data_load()
    if x is None:
        return
    
    # Data split
    x_train, x_val, X_test, y1_train, y1_val, y1_test = data_split(x, y1)
    _, _, _, y2_train, y2_val, y2_test = data_split(x, y2)
    
    # Data normalization
    x_train_norm, x_val_norm, xtest_normalize = data_normalization(x_train, x_val, X_test)
    
    # Data Training
    for t_name, y_train, y_val, y_test in [
        ('Y1 (Heating Load)', y1_train, y1_val, y1_test),
        ('Y2 (Cooling Load)', y2_train, y2_val, y2_test)
        ]:
        print(f"\nTraining models for {t_name}")
        
        # Linear Regression
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(x_train_norm, y_train)
        linear_prediction = linear_regression_model.prediction(xtest_normalize)
        linear_metric_value = evaluate_model(y_test, linear_prediction)
        
        # Ridge Regression
        ridge_regression_model = RidgeRegression(strength=1.0)
        ridge_regression_model.fit(x_train_norm, y_train)
        ridge_prediction = ridge_regression_model.prediction(xtest_normalize)
        ridge_metric_value = evaluate_model(y_test, ridge_prediction)
        
        print(f"\nResults for {t_name}:")
        print("Linear Regression Results:")
        for metric, value in linear_metric_value.items():
            print(f"  {metric}: {value:.5f}")
        
        print("Ridge Regression Results:")
        for metric, value in ridge_metric_value.items():
            print(f"  {metric}: {value:.5f}")

if __name__ == "__main__":
    main()
