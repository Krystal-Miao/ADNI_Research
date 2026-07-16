from pathlib import Path
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RandomIntensityAugmentation:
    """
    Apply mild MRI intensity changes without changing spatial anatomy.

    Operations:
    - random intensity scaling
    - random intensity shift
    - mild Gaussian noise
    """

    def __init__(
        self,
        scale_range=(0.98, 1.02),
        shift_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.0),
    ):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.noise_std_range = noise_std_range

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(*self.scale_range)
        shift = random.uniform(*self.shift_range)
        noise_std = random.uniform(*self.noise_std_range)

        image = image * scale + shift

        if noise_std > 0:
            image = image + torch.randn_like(image) * noise_std

        return torch.clamp(image, 0.0, 1.0)


class ADNISliceDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        augment=False,
        intensity_augment=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if augment and intensity_augment:
            raise ValueError(
                "Use either geometric augmentation or intensity augmentation, "
                "not both in the same experiment."
            )

        transform_steps = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
        ]

        # Existing geometric augmentation
        if augment:
            transform_steps.extend([
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
            ])

        transform_steps.append(transforms.ToTensor())

        # New intensity augmentation
        if intensity_augment:
            transform_steps.append(RandomIntensityAugmentation())

        transform_steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["slice_path"])
        label = int(row["label"])
        subject = row["subject"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            subject,
        )