import torch
from torch.utils.data import DataLoader
from dataset import ECGDataset
from model import ECGCNN

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


# Load dataset
test_dataset = ECGDataset("stage_2/heart_ecg/data/mitbih_test.csv")

test_loader = DataLoader(test_dataset, batch_size=64)


# Load trained model
model = ECGCNN()
model.load_state_dict(torch.load("stage_2/heart_ecg/ecg_model.pth"))

model.eval()


all_preds = []
all_labels = []
all_probs = []


with torch.no_grad():

    for X, y in test_loader:

        outputs = model(X).squeeze()

        probs = outputs.numpy()

        preds = (outputs > 0.5).float().numpy()

        all_preds.extend(preds)
        all_labels.extend(y.numpy())
        all_probs.extend(probs)


# Convert to arrays
import numpy as np

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# Metrics
accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(all_labels, all_preds)

recall = recall_score(all_labels, all_preds)

f1 = f1_score(all_labels, all_preds)

roc_auc = roc_auc_score(all_labels, all_probs)

cm = confusion_matrix(all_labels, all_preds)


print("\n===== ECG MODEL EVALUATION =====\n")

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC AUC    : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)