import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/processed/heart/heart_processed.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BINS = 10


def main():
    print("\n===== HEART CALIBRATION =====")

    # --------------------------
    # 1️⃣ Load Dataset
    # --------------------------
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Same deterministic split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # --------------------------
    # 2️⃣ Load Trained Model
    # --------------------------
    model = joblib.load("models/heart_gb.pkl")

    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------
    # 3️⃣ Brier Score
    # --------------------------
    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier Score: {brier:.4f}")

    # --------------------------
    # 4️⃣ Calibration Curve
    # --------------------------
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=N_BINS,
        strategy="uniform"
    )

    print("\nCalibration bins:")
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"Predicted={p_pred:.3f} | Actual={p_true:.3f}")


if __name__ == "__main__":
    main()
