import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file, header=None)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Convert to binary classification
        y = (y != 0).astype(int)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Add channel dimension for CNN
        self.X = self.X.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]