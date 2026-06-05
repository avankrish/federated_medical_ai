import pandas as pd
import joblib


def generate_heart_logits():
    public_df = pd.read_csv("fedmd/public_data.csv")

    heart_features = ["age", "trestbps", "chol"]
    X_public_heart = public_df[heart_features]

    model = joblib.load("models/heart_gb.pkl")
    logits = model.predict_proba(X_public_heart)[:, 1]

    pd.DataFrame({"heart_logit": logits}).to_csv(
        "fedmd/logits/heart_logits.csv", index=False
    )

    print("✅ Heart logits generated successfully.")


if __name__ == "__main__":
    generate_heart_logits()
