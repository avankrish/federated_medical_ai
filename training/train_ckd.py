import pandas as pd
import numpy as np
from pathlib import Path
import joblib

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#pytorch
import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH='data/processed/ckd/ckd_processed.csv'
RESULTS_PATH='evaluation/results/ckd_models.csv'

TEST_SIZE=0.2
RANDOM_STATE=42
EPOCHS=50
LEARNING_RATE=0.001

def log_results(model_name,acc,prec,rec,f1,notes=""):
    Path("evaluation/results").mkdir(parents=True,exist_ok=True)

    row=pd.DataFrame([{
        "model":model_name,
        "accuracy":acc,
        "precision":prec,
        "recall":rec,
        "f1_score":f1,
        "notes":notes
    }])

    if Path(RESULTS_PATH).exists():
        row.to_csv(RESULTS_PATH,mode='a',header=False,index=False)
    else:
        row.to_csv(RESULTS_PATH,index=False)


#MLP MODEL 
class CKDMLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def  forward(self,x):
        return self.net(x)
    
def main():
    df=pd.read_csv(DATA_PATH)

    X=df.drop(columns=['prognosis'])
    y=df['prognosis']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)

    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    joblib.dump(scaler, "models/ckd_scaler.pkl")

    #Logistic regression experimental
    lr=LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled,y_train)

    y_pred_lr=lr.predict(X_test_scaled)

    acc=accuracy_score(y_test,y_pred_lr)
    prec=precision_score(y_test,y_pred_lr)
    rec=recall_score(y_test,y_pred_lr)
    f1=f1_score(y_test,y_pred_lr)

    log_results(model_name='Logistic Regression',
                acc=acc,
                rec=rec,
                prec=prec,f1=f1,
                notes="Baseine Linear Model")
    
    print("Logistic Regression Results:")
    print(f"Accuracy:{acc:4f},F1 : {f1:4f}")

 #MLP final model
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    model = CKDMLP(input_dim=X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- TRAIN LOOP (NO EVALUATION HERE) ----
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f}")

    # ---- FINAL EVALUATION (ONCE) ----
    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_t).squeeze()
        y_pred_nn = (y_pred_nn >= 0.5).float()

    acc = accuracy_score(y_test_t, y_pred_nn)
    prec = precision_score(y_test_t, y_pred_nn, zero_division=0)
    rec = recall_score(y_test_t, y_pred_nn, zero_division=0)
    f1 = f1_score(y_test_t, y_pred_nn, zero_division=0)

    log_results(
        model_name="MLP",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        notes="Captures non-linear interactions (eGFR + albumin)"
    )
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ckd_mlp.pth")

    print("\nMLP Final Results:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()