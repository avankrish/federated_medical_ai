'''import pandas as pd
import numpy as np
from pathlib import Path


def create_public_dataset():
    np.random.seed(42)
    n = 300
    age_values = np.random.randint(18, 80, size=n)
    public_df = pd.DataFrame({
        # Shared
        "Age": age_values,
        "age": age_values,

        # CKD proxies
        "egfr": np.full(n, 90.0),
        "al": np.zeros(n),
        "sc": np.full(n, 1.0),

        # Diabetes proxies
        "Glucose": np.full(n, 100.0),
        "BMI": np.full(n, 25.0),
        "Insulin": np.full(n, 80.0),

        # Heart proxies (ONLY features actually used)
        "trestbps": np.full(n, 120.0),
        "chol": np.full(n, 200.0),
    })

    Path("fedmd/logits").mkdir(parents=True, exist_ok=True)
    public_df.to_csv("fedmd/public_data.csv", index=False)

    print("Public dataset created:", public_df.shape)
    print(public_df.head())


if __name__ == "__main__":
    create_public_dataset()
'''

import pandas as pd
import numpy as np
from pathlib import Path


def create_public_dataset():
    np.random.seed(42)
    n = 500   # more samples = better distillation

    public_df = pd.DataFrame({

        # =========================
        # AGE (shared)
        # =========================
        "Age": np.random.randint(18, 85, size=n),
        "age": np.random.randint(18, 85, size=n),

        # =========================
        # CKD FEATURES
        # =========================
        "egfr": np.random.normal(75, 25, n),   # CKD severity range
        "al": np.random.randint(0, 4, n),      # Albumin levels
        "sc": np.random.normal(1.3, 0.6, n),   # Creatinine

        # =========================
        # DIABETES FEATURES
        # =========================
        "Glucose": np.random.normal(130, 40, n),
        "BMI": np.random.normal(28, 6, n),
        "Insulin": np.random.normal(110, 60, n),

        # =========================
        # HEART FEATURES
        # =========================
        "trestbps": np.random.normal(130, 25, n),
        "chol": np.random.normal(220, 50, n),
    })

    # --------- Clip to realistic ranges ---------

    public_df["egfr"] = public_df["egfr"].clip(5, 130)
    public_df["sc"] = public_df["sc"].clip(0.4, 6)

    public_df["Glucose"] = public_df["Glucose"].clip(60, 300)
    public_df["BMI"] = public_df["BMI"].clip(15, 60)
    public_df["Insulin"] = public_df["Insulin"].clip(5, 400)

    public_df["trestbps"] = public_df["trestbps"].clip(80, 220)
    public_df["chol"] = public_df["chol"].clip(100, 500)

    Path("fedmd").mkdir(exist_ok=True)
    public_df.to_csv("fedmd/public_data.csv", index=False)

    print("✅ New public dataset created:", public_df.shape)
    print(public_df.describe())


if __name__ == "__main__":
    create_public_dataset()