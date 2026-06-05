import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# ---- CKD MLP Definition ----
class CKDMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---- CONFIG ----
DATA_PATH = "data/processed/ckd/ckd_processed.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def calibrate_ckd():

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = joblib.load("models/ckd_scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = CKDMLP(input_dim=X_test_t.shape[1])
    model.load_state_dict(torch.load("models/ckd_mlp.pth"))
    model.eval()

    with torch.no_grad():
        y_prob = model(X_test_t).squeeze().numpy()

    # ---- Brier Score ----
    brier = brier_score_loss(y_test, y_prob)

    print("\n===== CKD CALIBRATION =====")
    print(f"Brier Score: {brier:.4f}")

    # ---- Reliability Data ----
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    print("\nCalibration bins:")
    for p_t, p_p in zip(prob_true, prob_pred):
        print(f"Predicted={p_p:.3f} | Actual={p_t:.3f}")

if __name__ == "__main__":
    calibrate_ckd()
