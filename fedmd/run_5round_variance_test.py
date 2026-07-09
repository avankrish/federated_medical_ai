

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "inference"))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

from inference.test_scenarios import test_cases

N_ROUNDS = 5
N_PUBLIC_SAMPLES = 500

CKD_THRESHOLD = 0.55
DIABETES_THRESHOLD = 0.5
HEART_THRESHOLD = 0.6


# ----------------------------
# Model definitions
# ----------------------------
class CKDMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class FedMDStudent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def make_fresh_public_dataset(n=N_PUBLIC_SAMPLES):
    """Same distribution/clipping as fedmd/public_dataset.py,
    but UNSEEDED so each trial draws an independently fresh sample."""
    public_df = pd.DataFrame({
        "Age":      np.random.randint(18, 85, size=n),
        "age":      np.random.randint(18, 85, size=n),
        "egfr":     np.random.normal(75, 25, n),
        "al":       np.random.randint(0, 4, n),
        "sc":       np.random.normal(1.3, 0.6, n),
        "Glucose":  np.random.normal(130, 40, n),
        "BMI":      np.random.normal(28, 6, n),
        "Insulin":  np.random.normal(110, 60, n),
        "trestbps": np.random.normal(130, 25, n),
        "chol":     np.random.normal(220, 50, n),
    })
    public_df["egfr"]     = public_df["egfr"].clip(5, 130)
    public_df["sc"]       = public_df["sc"].clip(0.4, 6)
    public_df["Glucose"]  = public_df["Glucose"].clip(60, 300)
    public_df["BMI"]      = public_df["BMI"].clip(15, 60)
    public_df["Insulin"]  = public_df["Insulin"].clip(5, 400)
    public_df["trestbps"] = public_df["trestbps"].clip(80, 220)
    public_df["chol"]     = public_df["chol"].clip(100, 500)
    return public_df


def generate_logits(public_df):
    """Run all three FIXED pretrained teachers on the fresh public_df."""

    # ---- CKD (MLP) ----
    ckd_scaler = joblib.load("models/ckd_scaler.pkl")
    ckd_features = list(ckd_scaler.feature_names_in_)
    X_ckd = ckd_scaler.transform(public_df[ckd_features])
    X_ckd_t = torch.tensor(X_ckd, dtype=torch.float32)

    ckd_model = CKDMLP(input_dim=X_ckd_t.shape[1])
    ckd_model.load_state_dict(torch.load("models/ckd_mlp.pth"))
    ckd_model.eval()
    with torch.no_grad():
        ckd_logits = ckd_model(X_ckd_t).squeeze().numpy()

    # ---- Diabetes (LogReg) ----
    dia_scaler = joblib.load("models/diabetes_scaler.pkl")
    dia_features = list(dia_scaler.feature_names_in_)
    X_dia = dia_scaler.transform(public_df[dia_features])
    dia_model = joblib.load("models/diabetes_logreg.pkl")
    dia_logits = dia_model.predict_proba(X_dia)[:, 1]

    # ---- Heart (GradBoost) ----
    heart_features = ["age", "trestbps", "chol"]
    X_heart = public_df[heart_features]
    heart_model = joblib.load("models/heart_gb.pkl")
    heart_logits = heart_model.predict_proba(X_heart)[:, 1]

    agg_df = pd.DataFrame({
        "ckd_soft":      ckd_logits,
        "diabetes_soft": dia_logits,
        "heart_soft":    heart_logits,
    })
    return agg_df


def train_fresh_student(public_df, agg_df, epochs=200, lr=0.001):
    X_t = torch.tensor(public_df.values, dtype=torch.float32)
    y_t = torch.tensor(agg_df.values, dtype=torch.float32)

    model = FedMDStudent(input_dim=X_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model, loss.item()


def evaluate_student(model, public_schema):
    y_true, y_pred = [], []

    for name, input_data, expected in test_cases:
        fedmd_input = {col: input_data.get(col, 0) for col in public_schema}
        X = torch.tensor(
            pd.DataFrame([fedmd_input]).values, dtype=torch.float32
        )
        with torch.no_grad():
            probs = model(X).numpy()[0]

        predicted = [
            int(probs[0] >= CKD_THRESHOLD),
            int(probs[1] >= DIABETES_THRESHOLD),
            int(probs[2] >= HEART_THRESHOLD),
        ]
        y_true.append(expected)
        y_pred.append(predicted)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    exact_match = (y_true == y_pred).all(axis=1).mean()
    hamming     = (y_true == y_pred).mean()
    f1_micro    = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_disease = {}
    for idx, dname in enumerate(["CKD", "Diabetes", "Heart"]):
        per_disease[dname] = {
            "acc": (y_true[:, idx] == y_pred[:, idx]).mean(),
            "f1":  f1_score(
                y_true[:, idx], y_pred[:, idx], zero_division=0
            ),
        }

    return {
        "exact_match": exact_match,
        "hamming":     hamming,
        "f1_micro":    f1_micro,
        "f1_macro":    f1_macro,
        "per_disease": per_disease,
    }


def main():
    results = []

    for round_idx in range(1, N_ROUNDS + 1):
        print(f"\n{'='*60}\nTRIAL {round_idx}/{N_ROUNDS}\n{'='*60}")

        public_df            = make_fresh_public_dataset()
        agg_df               = generate_logits(public_df)
        student, final_loss  = train_fresh_student(public_df, agg_df)
        metrics              = evaluate_student(
            student, public_df.columns.tolist()
        )
        metrics["round"]           = round_idx
        metrics["final_train_loss"] = final_loss
        results.append(metrics)

        print(f"Final student train MSE loss : {final_loss:.4f}")
        print(
            f"Exact Match : {metrics['exact_match']:.2%} | "
            f"Hamming : {metrics['hamming']:.2%} | "
            f"F1_micro : {metrics['f1_micro']:.3f} | "
            f"F1_macro : {metrics['f1_macro']:.3f}"
        )
        for dname, d in metrics["per_disease"].items():
            print(
                f"  {dname:10s} acc={d['acc']:.2%}  f1={d['f1']:.3f}"
            )

    # ---- Summary ----
    print(
        f"\n{'='*60}\n"
        f"SUMMARY ACROSS {N_ROUNDS} TRIALS (mean +/- std)\n"
        f"{'='*60}"
    )

    em   = np.array([r["exact_match"] for r in results])
    ha   = np.array([r["hamming"]     for r in results])
    f1mi = np.array([r["f1_micro"]    for r in results])
    f1ma = np.array([r["f1_macro"]    for r in results])

    print(
        f"Exact Match Accuracy : "
        f"{em.mean():.2%} +/- {em.std():.2%}  "
        f"(range {em.min():.2%}-{em.max():.2%})"
    )
    print(
        f"Hamming Accuracy     : "
        f"{ha.mean():.2%} +/- {ha.std():.2%}  "
        f"(range {ha.min():.2%}-{ha.max():.2%})"
    )
    print(f"F1 Micro             : {f1mi.mean():.3f} +/- {f1mi.std():.3f}")
    print(f"F1 Macro             : {f1ma.mean():.3f} +/- {f1ma.std():.3f}")

    print("\nPer-disease (mean +/- std across trials):")
    for dname in ["CKD", "Diabetes", "Heart"]:
        accs = np.array([r["per_disease"][dname]["acc"] for r in results])
        f1s  = np.array([r["per_disease"][dname]["f1"]  for r in results])
        print(
            f"  {dname:10s} "
            f"acc = {accs.mean():.2%} +/- {accs.std():.2%}   "
            f"f1 = {f1s.mean():.3f} +/- {f1s.std():.3f}"
        )

    # ---- Save CSV ----
    rows = []
    for r in results:
        row = {
            "trial":            r["round"],
            "exact_match":      r["exact_match"],
            "hamming":          r["hamming"],
            "f1_micro":         r["f1_micro"],
            "f1_macro":         r["f1_macro"],
            "final_train_loss": r["final_train_loss"],
        }
        for dname in ["CKD", "Diabetes", "Heart"]:
            row[f"{dname}_acc"] = r["per_disease"][dname]["acc"]
            row[f"{dname}_f1"]  = r["per_disease"][dname]["f1"]
        rows.append(row)

    out_df  = pd.DataFrame(rows)
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation", "results",
        "fedmd_5round_variance.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-trial results to {out_path}")


if __name__ == "__main__":
    main()
