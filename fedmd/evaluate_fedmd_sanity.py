import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# -----------------------------
# FedMD Student Model (same as training)
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


def evaluate_fedmd_sanity():

    print("\n===== FEDMD SANITY EVALUATION =====\n")

    # -----------------------------
    # Load public dataset
    # -----------------------------
    public_df = pd.read_csv("fedmd/public_data.csv")
    X_public = torch.tensor(public_df.values, dtype=torch.float32)

    # -----------------------------
    # Load FedMD student
    # -----------------------------
    model = FedMDStudent(input_dim=X_public.shape[1])
    model.load_state_dict(torch.load("fedmd/models/fedmd_student.pth"))
    model.eval()

    # -----------------------------
    # Run FedMD on public data
    # -----------------------------
    with torch.no_grad():
        fedmd_outputs = model(X_public).numpy()

    fedmd_df = pd.DataFrame(
        fedmd_outputs,
        columns=["CKD_prob", "Diabetes_prob", "Heart_prob"]
    )

    # -----------------------------
    # CHECK 1: Range & stability
    # -----------------------------
    print("▶ CHECK 1: Probability Range & Stability\n")
    print(fedmd_df.describe(), "\n")

    # -----------------------------
    # CHECK 2: Neutrality check
    # -----------------------------
    print("▶ CHECK 2: Neutrality on Public Data\n")
    print("Max probabilities:")
    print(fedmd_df.max(), "\n")

    # -----------------------------
    # CHECK 3: Agreement trend with client logits
    # -----------------------------
    print("▶ CHECK 3: Agreement Trend with Client Logits\n")

    ckd_logits = pd.read_csv("fedmd/logits/ckd_logits.csv")
    diabetes_logits = pd.read_csv("fedmd/logits/diabetes_logits.csv")
    heart_logits = pd.read_csv("fedmd/logits/heart_logits.csv")

    agreement_df = pd.DataFrame({
        "CKD_client": ckd_logits.iloc[:, 0],
        "CKD_fedmd": fedmd_df["CKD_prob"],
        "Diabetes_client": diabetes_logits.iloc[:, 0],
        "Diabetes_fedmd": fedmd_df["Diabetes_prob"],
        "Heart_client": heart_logits.iloc[:, 0],
        "Heart_fedmd": fedmd_df["Heart_prob"]
    })

    print(agreement_df.corr(), "\n")

    # -----------------------------
    # CHECK 4: Overconfidence check
    # -----------------------------
    print("▶ CHECK 4: Overconfidence Check\n")

    high_conf = (fedmd_df > 0.8).sum()
    print("Number of samples with probability > 0.8:")
    print(high_conf)

    print("\n===== FEDMD SANITY EVALUATION COMPLETE =====\n")


if __name__ == "__main__":
    evaluate_fedmd_sanity()
