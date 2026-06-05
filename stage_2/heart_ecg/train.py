import torch
from torch.utils.data import DataLoader
from dataset import ECGDataset
from model import ECGCNN


train_dataset = ECGDataset("stage_2/heart_ecg/data/mitbih_train.csv")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = ECGCNN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for X, y in train_loader:

        optimizer.zero_grad()

        outputs = model(X).squeeze()

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "stage_2/heart_ecg/ecg_model.pth")

print("Model saved.")