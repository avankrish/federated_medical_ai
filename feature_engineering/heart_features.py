import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/heart/heart_disease_uci.csv"
PROCESSED_PATH = "data/processed/heart/heart_processed.csv"


def preprocess_heart():
    df = pd.read_csv(RAW_PATH)

   
    df = df[['age', 'trestbps', 'chol', 'prognosis']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(), inplace=True)

    Path("data/processed/heart").mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Heart preprocessing complete.")


if __name__ == "__main__":
    preprocess_heart()
