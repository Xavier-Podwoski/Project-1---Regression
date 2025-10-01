# Project-1---Regression

Energy Efficiency Regression
============================

Code Features:
Linear Regression:
   Uses the Normal Equation:
      w = (X^T * X)^-1 * X^T * y

Ridge Regression:
   Adds L2 for overfitting prevention/reduction:
      w = (X^T * X + lamda(I))^-1 * X^T * y

Metrics Implemeneted and Solved:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R^2 Score

Make sure that the dataset from UCI Machine Learning Repository – Energy Efficiency is downloaded

How to run:
1. Create a Python virtual environment:
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r required_installs.txt

3. Run the pipeline (downloads dataset from UCI):
   make run

If you don’t want auto-download, manually place ENB2012_data.xlsx into `data/` and run:
   python3 run_regression.py --no-download


