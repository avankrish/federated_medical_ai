import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np


# -----------------------------
# FedMD Student Model
# -----------------------------
class FedMDStudent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# CKD MLP Model
# -----------------------------
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


# -----------------------------
# Utility: check feature availability
# -----------------------------
def has_features(user_input, required_features):
    return all(f in user_input and user_input[f] is not None for f in required_features)


# -----------------------------
# Stage-1 Inference
# -----------------------------
def stage1_inference(user_input):
    CKD_THRESHOLD = 0.55
    DIABETES_THRESHOLD = 0.5
    HEART_THRESHOLD = 0.6
    results = {"CKD": 0, "Diabetes": 0, "Heart": 0}
    probs = {}

    # =====================
    # CKD CLIENT MODEL
    # =====================
    ckd_features = ["sc", "egfr", "al", "age"]

    if has_features(user_input, ckd_features):
        scaler = joblib.load("models/ckd_scaler.pkl")

        # ✅ FIRST create input
        X_ckd = pd.DataFrame([user_input])[scaler.feature_names_in_]
        X_ckd = torch.tensor(scaler.transform(X_ckd), dtype=torch.float32)

        # ✅ THEN create model
        model = CKDMLP(input_dim=X_ckd.shape[1])
        model.load_state_dict(torch.load("models/ckd_mlp.pth"))
        model.eval()

        with torch.no_grad():
            p_ckd = model(X_ckd).item()

        probs["CKD"] = p_ckd
        results["CKD"] = int(p_ckd >= CKD_THRESHOLD)

    # =====================
    # DIABETES CLIENT MODEL
    # =====================
    scaler = joblib.load("models/diabetes_scaler.pkl")
    diabetes_features = list(scaler.feature_names_in_)
    if has_features(user_input, diabetes_features):
        model = joblib.load("models/diabetes_logreg.pkl")

        X_diabetes = pd.DataFrame([user_input])[scaler.feature_names_in_]
        X_diabetes = scaler.transform(X_diabetes)

        p_diabetes = model.predict_proba(X_diabetes)[0, 1]
        probs["Diabetes"] = p_diabetes
        results["Diabetes"] = int(p_diabetes >= DIABETES_THRESHOLD)

    # =====================
    # HEART CLIENT MODEL
    # =====================
    heart_features = ["age", "trestbps", "chol"]

    if has_features(user_input, heart_features):
        model = joblib.load("models/heart_gb.pkl")

        X_heart = pd.DataFrame([user_input])[heart_features]
        p_heart = model.predict_proba(X_heart)[0, 1]

        probs["Heart"] = p_heart
        results["Heart"] = int(p_heart >= HEART_THRESHOLD)

    # =====================
    # FedMD COORDINATION (ADVISORY)
    # =====================
    try:
        public_schema = pd.read_csv("fedmd/public_data.csv").columns.tolist()
        fedmd_input = {col: user_input.get(col, 0) for col in public_schema}

        X_fedmd = torch.tensor(pd.DataFrame([fedmd_input]).values, dtype=torch.float32)

        fedmd_model = FedMDStudent(input_dim=X_fedmd.shape[1])
        fedmd_model.load_state_dict(torch.load("fedmd/models/fedmd_student.pth"))
        fedmd_model.eval()

        with torch.no_grad():
            fedmd_probs = fedmd_model(X_fedmd).numpy()[0]

        # Upgrade borderline heart cases
        if (
                "Heart" in probs
                and 0.45<= probs["Heart"] < HEART_THRESHOLD
                and fedmd_probs[2] > 0.6
            ):
            results["Heart"] = 1

    except Exception:
        pass  # FedMD is advisory only

    return {
        "probabilities": probs,
        "final_output": [results["CKD"], results["Diabetes"], results["Heart"]]
    }