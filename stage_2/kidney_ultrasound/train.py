import os
import torch
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


# train / test split

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.15,
    random_state=42
)

train_dataset = KidneyDataset(X_train, y_train)
test_dataset = KidneyDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


model = KidneyCNN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):

    total_loss = 0

    for X, y in train_loader:

        optimizer.zero_grad()

        outputs = model(X).squeeze()

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")


torch.save(model.state_dict(), "stage_2/kidney_ultrasound/kidney_model.pth")

print("Kidney model saved")