# Project-1---Regression

Energy Efficiency Regression
============================

Contents:
- run_regression.py: main script (download dataset, train models, evaluate, produce report.pdf in outputs/)
- requirements.txt: Python dependencies
- data/: where the dataset will be downloaded (or place ENB2012_data.xlsx here)
- outputs/: generated outputs (metrics.json, models/, report.pdf, results_test.csv)

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

Outputs:
- `outputs/metrics.json` : evaluation metrics
- `outputs/results_test.csv` : test predictions
- `outputs/models/` : trained models
