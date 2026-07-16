import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ADNISubjectDataset(Dataset):
    """
    Returns all MRI slices belonging to one subject.

    Output:
        images:  Tensor with shape [num_slices, 3, 224, 224]
        label:   Scalar tensor
        subject: Subject ID string
    """

    def __init__(
        self,
        csv_path,
        split,
        expected_slices=30,
    ):
        dataframe = pd.read_csv(csv_path)

        dataframe = dataframe[
            dataframe["split"] == split
        ].copy()

        dataframe["slice_index"] = dataframe["slice_path"].apply(
            self.extract_slice_index
        )

        self.expected_slices = expected_slices

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.subject_records = []

        for subject, subject_dataframe in dataframe.groupby("subject"):
            subject_dataframe = subject_dataframe.sort_values(
                "slice_index"
            ).reset_index(drop=True)

            labels = subject_dataframe["label"].unique()

            if len(labels) != 1:
                raise ValueError(
                    f"Subject {subject} has inconsistent labels: "
                    f"{labels.tolist()}"
                )

            if len(subject_dataframe) != self.expected_slices:
                raise ValueError(
                    f"Subject {subject} has "
                    f"{len(subject_dataframe)} slices; "
                    f"expected {self.expected_slices}."
                )

            slice_paths = [
                Path(path)
                for path in subject_dataframe["slice_path"].tolist()
            ]

            slice_indices = (
                subject_dataframe["slice_index"]
                .astype(int)
                .tolist()
            )

            expected_indices = list(range(self.expected_slices))

            if slice_indices != expected_indices:
                raise ValueError(
                    f"Subject {subject} has unexpected slice indices: "
                    f"{slice_indices}"
                )

            self.subject_records.append({
                "subject": str(subject),
                "label": int(labels[0]),
                "slice_paths": slice_paths,
            })

        self.subject_records.sort(
            key=lambda record: record["subject"]
        )

    @staticmethod
    def extract_slice_index(slice_path):
        """
        Extract the integer N from a filename ending in _sliceN.tiff.
        """

        match = re.search(
            r"_slice(\d+)\.tiff$",
            str(slice_path),
        )

        if match is None:
            raise ValueError(
                f"Could not extract slice index from: {slice_path}"
            )

        return int(match.group(1))

    def __len__(self):
        return len(self.subject_records)

    def __getitem__(self, index):
        record = self.subject_records[index]

        images = []

        for image_path in record["slice_paths"]:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(image_path).convert("L")
            image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)

        label = torch.tensor(
            record["label"],
            dtype=torch.long,
        )

        return images, label, record["subject"]