import pandas as pd
from pathlib import Path
import joblib

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/processed/heart/heart_processed.csv"
RESULTS_PATH = "evaluation/results/heart_models.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# -----------------------------
# HELPER: LOG RESULTS
# -----------------------------
def log_results(model_name, acc, prec, rec, f1, notes=""):
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "notes": notes
    }])

    if Path(RESULTS_PATH).exists():
        row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(RESULTS_PATH, index=False)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # =============================
    # MODEL 1: LOGISTIC REGRESSION experimental 
    # =============================
    # Scaling required
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_lr)
    prec = precision_score(y_test, y_pred_lr, zero_division=0)
    rec = recall_score(y_test, y_pred_lr, zero_division=0)
    f1 = f1_score(y_test, y_pred_lr, zero_division=0)

    log_results(
        model_name="Logistic Regression",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        notes="Linear baseline; limited ability to capture complex interactions"
    )

    print("\nLogistic Regression Results:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # =============================
    # MODEL 2: RANDOM FOREST experimental
    # =============================
    # No scaling needed
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf, zero_division=0)
    rec = recall_score(y_test, y_pred_rf, zero_division=0)
    f1 = f1_score(y_test, y_pred_rf, zero_division=0)

    log_results(
        model_name="Random Forest",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        notes="Captures non-linear risk factor interactions; robust tabular model"
    )

    print("\nRandom Forest Results:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # =============================
    # MODEL 3: GRADIENT BOOSTING final model
    # =============================
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )

    gb.fit(X_train, y_train)
    joblib.dump(gb, "models/heart_gb.pkl")

    y_pred_gb = gb.predict(X_test)

    acc = accuracy_score(y_test, y_pred_gb)
    prec = precision_score(y_test, y_pred_gb, zero_division=0)
    rec = recall_score(y_test, y_pred_gb, zero_division=0)
    f1 = f1_score(y_test, y_pred_gb, zero_division=0)

    log_results(
        model_name="Gradient Boosting",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        notes="Boosted ensemble capturing complex non-linear risk patterns"
    )

    print("\nGradient Boosting Results:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
