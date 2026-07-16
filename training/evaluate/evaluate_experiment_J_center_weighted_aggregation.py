import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CSV_PATH = "data/slice_metadata_split.csv"

MODEL_PATH = (
    "training/checkpoints/"
    "best_experiment_C_resnet18_layer4.pt"
)

BATCH_SIZE = 16

# There are 30 central slices: slice0 through slice29.
NUMBER_OF_SLICES = 30

# Controls how strongly the middle slices are emphasized.
# A larger value produces weights closer to ordinary averaging.
GAUSSIAN_SIGMA = 7.5


class ADNISliceDatasetWithIndex(Dataset):
    """
    Loads each MRI slice and also returns its slice index.

    Example filename:
        ..._slice14.tiff

    Returned values:
        image, label, subject, slice_index
    """

    def __init__(self, csv_path, split):
        dataframe = pd.read_csv(csv_path)

        self.dataframe = dataframe[
            dataframe["split"] == split
        ].reset_index(drop=True)

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
        return len(self.dataframe)

    @staticmethod
    def extract_slice_index(slice_path):
        """
        Extract the number from a filename ending in _sliceN.tiff.
        """

        match = re.search(r"_slice(\d+)\.tiff$", str(slice_path))

        if match is None:
            raise ValueError(
                f"Could not extract slice index from: {slice_path}"
            )

        return int(match.group(1))

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["slice_path"])
        label = int(row["label"])
        subject = str(row["subject"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        slice_index = self.extract_slice_index(image_path)

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            subject,
            slice_index,
        )


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model


def create_center_weights(
    number_of_slices=NUMBER_OF_SLICES,
    sigma=GAUSSIAN_SIGMA,
):
    """
    Create Gaussian weights centered between slices 14 and 15.

    Middle slices receive larger weights, while slices near the
    boundaries receive smaller weights.
    """

    slice_indices = np.arange(number_of_slices, dtype=np.float64)
    center = (number_of_slices - 1) / 2

    weights = np.exp(
        -0.5 * ((slice_indices - center) / sigma) ** 2
    )

    weights = weights / weights.sum()

    return weights


def collect_subject_slice_probabilities(model, loader, device):
    """
    Generate an AD probability for every slice and group the
    probabilities by subject.
    """

    model.eval()

    subject_slices = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():
        for images, labels, subjects, slice_indices in loader:
            images = images.to(device)

            outputs = model(images)
            ad_probabilities = torch.softmax(outputs, dim=1)[:, 1]

            for subject, label, probability, slice_index in zip(
                subjects,
                labels,
                ad_probabilities.cpu(),
                slice_indices,
            ):
                subject_slices[subject].append(
                    (
                        int(slice_index.item()),
                        float(probability.item()),
                    )
                )

                subject_labels[subject] = int(label.item())

    return subject_slices, subject_labels


def aggregate_mean(subject_slices):
    """
    Existing aggregation method:
    all slices contribute equally.
    """

    subject_probabilities = {}

    for subject, slice_information in subject_slices.items():
        probabilities = [
            probability
            for _, probability in slice_information
        ]

        subject_probabilities[subject] = float(
            np.mean(probabilities)
        )

    return subject_probabilities


def aggregate_center_weighted(subject_slices, center_weights):
    """
    Experiment J aggregation:
    middle slices contribute more than boundary slices.
    """

    subject_probabilities = {}

    for subject, slice_information in subject_slices.items():
        weighted_probability_sum = 0.0
        subject_weight_sum = 0.0

        for slice_index, probability in slice_information:
            if not 0 <= slice_index < len(center_weights):
                raise ValueError(
                    f"Unexpected slice index {slice_index} "
                    f"for subject {subject}"
                )

            weight = center_weights[slice_index]

            weighted_probability_sum += weight * probability
            subject_weight_sum += weight

        if subject_weight_sum == 0:
            raise RuntimeError(
                f"Total slice weight is zero for subject {subject}"
            )

        subject_probabilities[subject] = (
            weighted_probability_sum / subject_weight_sum
        )

    return subject_probabilities


def calculate_metrics(
    subject_probabilities,
    subject_labels,
    threshold,
):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for subject, probability in subject_probabilities.items():
        prediction = 1 if probability >= threshold else 0
        label = subject_labels[subject]

        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        else:
            false_negative += 1

    total = (
        true_positive
        + false_positive
        + true_negative
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )

    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def select_validation_threshold(
    subject_probabilities,
    subject_labels,
):
    """
    Select the threshold using validation balanced accuracy only.

    Tie-breaking:
    1. Higher balanced accuracy
    2. Higher sensitivity
    3. Threshold closer to 0.50
    """

    best_threshold = None
    best_metrics = None
    best_score = None

    for threshold in np.arange(0.05, 0.951, 0.01):
        threshold = round(float(threshold), 2)

        metrics = calculate_metrics(
            subject_probabilities,
            subject_labels,
            threshold,
        )

        score = (
            metrics["balanced_accuracy"],
            metrics["sensitivity"],
            -abs(threshold - 0.50),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def print_metrics(title, threshold, metrics):
    print(f"\n{title}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        "Confusion matrix: "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"TN={metrics['tn']} "
        f"FN={metrics['fn']}"
    )


def main():
    device = get_device()
    print("Device:", device)

    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model = create_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model = model.to(device)

    validation_dataset = ADNISliceDatasetWithIndex(
        CSV_PATH,
        split="val",
    )

    test_dataset = ADNISliceDatasetWithIndex(
        CSV_PATH,
        split="test",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    center_weights = create_center_weights()

    print("\nCenter weights:")
    for slice_index, weight in enumerate(center_weights):
        print(
            f"Slice {slice_index:02d}: "
            f"{weight:.4f}"
        )

    validation_slices, validation_labels = (
        collect_subject_slice_probabilities(
            model,
            validation_loader,
            device,
        )
    )

    test_slices, test_labels = (
        collect_subject_slice_probabilities(
            model,
            test_loader,
            device,
        )
    )

    # Existing Experiment C aggregation
    validation_mean_probabilities = aggregate_mean(
        validation_slices
    )

    test_mean_probabilities = aggregate_mean(
        test_slices
    )

    mean_threshold, mean_validation_metrics = (
        select_validation_threshold(
            validation_mean_probabilities,
            validation_labels,
        )
    )

    mean_test_metrics = calculate_metrics(
        test_mean_probabilities,
        test_labels,
        mean_threshold,
    )

    # Experiment J aggregation
    validation_weighted_probabilities = (
        aggregate_center_weighted(
            validation_slices,
            center_weights,
        )
    )

    test_weighted_probabilities = aggregate_center_weighted(
        test_slices,
        center_weights,
    )

    weighted_threshold, weighted_validation_metrics = (
        select_validation_threshold(
            validation_weighted_probabilities,
            validation_labels,
        )
    )

    weighted_test_metrics = calculate_metrics(
        test_weighted_probabilities,
        test_labels,
        weighted_threshold,
    )

    print("\n" + "=" * 72)
    print("Experiment C versus Experiment J")
    print("=" * 72)

    print_metrics(
        "Experiment C — Simple mean aggregation: validation",
        mean_threshold,
        mean_validation_metrics,
    )

    print_metrics(
        "Experiment C — Simple mean aggregation: test",
        mean_threshold,
        mean_test_metrics,
    )

    print_metrics(
        "Experiment J — Center-weighted aggregation: validation",
        weighted_threshold,
        weighted_validation_metrics,
    )

    print_metrics(
        "Experiment J — Center-weighted aggregation: test",
        weighted_threshold,
        weighted_test_metrics,
    )


if __name__ == "__main__":
    main()