from pathlib import Path

import pandas as pd

import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

class ADNISliceDataset(Dataset):

    def __init__(self, csv_path, split, augment=False):

        self.df = pd.read_csv(csv_path)

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if augment:

            self.transform = transforms.Compose([

                transforms.Grayscale(num_output_channels=3),

                transforms.Resize((224, 224)),

                transforms.RandomRotation(degrees=10),

                transforms.RandomAffine(

                    degrees=0,

                    translate=(0.05, 0.05),

                    scale=(0.95, 1.05),

                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225],

                ),

            ])

        else:

            self.transform = transforms.Compose([

                transforms.Grayscale(num_output_channels=3),

                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225],

                ),

            ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(row["slice_path"])

        label = int(row["label"])

        image = Image.open(image_path).convert("L")

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), row["subject"]