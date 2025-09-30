import pandas as pd

def load_data(filepath: str):
    df = pd.read_excel(filepath, engine="openpyxl")

X = df.iloc[:, 0:8]
Y = df.iloc[:, 8:10]

return X, Y
