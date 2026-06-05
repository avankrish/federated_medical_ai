import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/diabetes/Healthcare-Diabetes.csv"
PROCESSED_PATH = "data/processed/diabetes/diabetes_processed.csv"


def preprocess_diabetes():
    df = pd.read_csv(RAW_PATH)

    df = df[['Glucose', 'BMI', 'Insulin', 'Age', 'prognosis']]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    for col in ['Glucose', 'BMI', 'Insulin']:
        df[col] = df[col].replace(0, df[col].median())

    df.fillna(df.median(), inplace=True)

    # Save processed data
    Path("data/processed/diabetes").mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Diabetes preprocessing complete.")


if __name__ == "__main__":
    preprocess_diabetes()
