import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import KidneyDataset
from model import KidneyCNN


image_paths = []
labels = []

data_dir = "stage_2/kidney_ultrasound/data"

for folder in os.listdir(data_dir):

    folder_path = os.path.join(data_dir, folder)

    for img in os.listdir(folder_path):

        image_paths.append(os.path.join(folder_path, img))

        if folder.lower() == "normal":
            labels.append(0)
        else:
            labels.append(1)


# recreate the SAME split
X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.15,
    random_state=42
)

test_dataset = KidneyDataset(X_test, y_test)

loader = DataLoader(test_dataset, batch_size=32)


model = KidneyCNN()
model.load_state_dict(torch.load("stage_2/kidney_ultrasound/kidney_model.pth"))

model.eval()

y_true = []
y_pred = []
y_prob = []


with torch.no_grad():

    for X, y in loader:

        outputs = model(X).squeeze()

        preds = (outputs > 0.5).int()

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())
        y_prob.extend(outputs.numpy())


acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)


print("\n===== KIDNEY ULTRASOUND MODEL EVALUATION =====\n")

print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC AUC    : {roc:.4f}")

print("\nConfusion Matrix")
print(cm)