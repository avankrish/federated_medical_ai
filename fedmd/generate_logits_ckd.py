'''import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib


# ---- CKD MLP (same architecture) ----
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


def generate_ckd_logits():
    # Load public data
    public_df = pd.read_csv("fedmd/public_data.csv")

    # Load scaler + model (you should save these after training)
    scaler = joblib.load("models/ckd_scaler.pkl")

# 🔑 Get feature order used during training
    ckd_features = list(scaler.feature_names_in_)

    # Reorder public data to EXACT same order
    X_public_ckd = public_df[ckd_features]

    X_public = scaler.transform(X_public_ckd)


    X_public_t = torch.tensor(X_public, dtype=torch.float32)

    model = CKDMLP(input_dim=X_public_t.shape[1])
    model.load_state_dict(torch.load("models/ckd_mlp.pth"))
    model.eval()

    with torch.no_grad():
        logits = model(X_public_t).squeeze().numpy()

    pd.DataFrame({"ckd_logit": logits}).to_csv(
        "fedmd/logits/ckd_logits.csv", index=False
    )

    print("CKD logits generated.")


if __name__ == "__main__":
    generate_ckd_logits()
'''

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib


# ---- CKD MLP (same architecture) ----
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


def generate_ckd_logits():
    # Load public data
    public_df = pd.read_csv("fedmd/public_data.csv")

    # Load scaler + model (you should save these after training)
    scaler = joblib.load("models/ckd_scaler.pkl")

# 🔑 Get feature order used during training
    ckd_features = list(scaler.feature_names_in_)

    # Reorder public data to EXACT same order
    X_public_ckd = public_df[ckd_features]

    X_public = scaler.transform(X_public_ckd)


    X_public_t = torch.tensor(X_public, dtype=torch.float32)

    model = CKDMLP(input_dim=X_public_t.shape[1])
    model.load_state_dict(torch.load("models/ckd_mlp.pth"))
    model.eval()

    with torch.no_grad():
        logits = model(X_public_t).squeeze().numpy()

    pd.DataFrame({"ckd_logit": logits}).to_csv(
        "fedmd/logits/ckd_logits.csv", index=False
    )

    print("CKD logits generated.")


if __name__ == "__main__":
    generate_ckd_logits()