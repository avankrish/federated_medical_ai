'''import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


# ----------------------------
# Student Model
# ----------------------------
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


def train_student():
    # Load public data
    X = pd.read_csv("fedmd/public_data.csv")

    # Load aggregated logits
    y = pd.read_csv("fedmd/aggregated_logits.csv")[[
        "ckd_soft", "diabetes_soft", "heart_soft"
    ]]

    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)

    model = FedMDStudent(input_dim=X_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

    Path("fedmd/models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "fedmd/models/fedmd_student.pth")

    print("✅ FedMD student model trained and saved.")


if __name__ == "__main__":
    train_student()
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path



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


def train_student():
    # Load public data
    X = pd.read_csv("fedmd/public_data.csv")

    # Load aggregated logits
    y = pd.read_csv("fedmd/aggregated_logits.csv")[[
        "ckd_soft", "diabetes_soft", "heart_soft"
    ]]

    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)

    model = FedMDStudent(input_dim=X_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

    Path("fedmd/models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "fedmd/models/fedmd_student.pth")

    print("✅ FedMD student model trained and saved.")


if __name__ == "__main__":
    train_student()