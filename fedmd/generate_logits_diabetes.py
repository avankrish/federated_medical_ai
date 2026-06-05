'''import pandas as pd
import joblib


def generate_diabetes_logits():
    public_df = pd.read_csv("fedmd/public_data.csv")

    scaler = joblib.load("models/diabetes_scaler.pkl")
    diabetes_features = list(scaler.feature_names_in_)
    X_public_diabetes = public_df[diabetes_features]

    model = joblib.load("models/diabetes_logreg.pkl")

    X_public_scaled = scaler.transform(X_public_diabetes)
    logits = model.predict_proba(X_public_scaled)[:, 1]

    pd.DataFrame({"diabetes_logit": logits}).to_csv(
        "fedmd/logits/diabetes_logits.csv", index=False
    )

    print("✅ Diabetes logits generated successfully.")


if __name__ == "__main__":
    generate_diabetes_logits()
'''

import pandas as pd
import joblib


def generate_diabetes_logits():
    public_df = pd.read_csv("fedmd/public_data.csv")

    scaler = joblib.load("models/diabetes_scaler.pkl")
    diabetes_features = list(scaler.feature_names_in_)
    X_public_diabetes = public_df[diabetes_features]

    model = joblib.load("models/diabetes_logreg.pkl")

    X_public_scaled = scaler.transform(X_public_diabetes)
    logits = model.predict_proba(X_public_scaled)[:, 1]

    pd.DataFrame({"diabetes_logit": logits}).to_csv(
        "fedmd/logits/diabetes_logits.csv", index=False
    )

    print("✅ Diabetes logits generated successfully.")


if __name__ == "__main__":
    generate_diabetes_logits()