import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KidneyDataset(Dataset):

    def __init__(self, image_paths, labels):

        self.image_paths = image_paths
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label