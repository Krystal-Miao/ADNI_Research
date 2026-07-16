from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.mean_feature_model import ResNet18MeanFeaturePooling
from training.subject_dataset import ADNISubjectDataset


CSV_PATH = "data/slice_metadata_split.csv"

EXPERIMENT_C_CHECKPOINT = (
    "training/checkpoints/"
    "best_experiment_C_resnet18_layer4.pt"
)

EXPERIMENT_L_CHECKPOINT = (
    "training/checkpoints/"
    "best_experiment_L_resnet18_mean_feature_pooling.pt"
)

BATCH_SIZE = 2


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def collect_predictions(model, loader, device):
    model.eval()

    subject_probabilities = {}
    subject_labels = {}

    with torch.no_grad():
        for images, labels, subjects in loader:
            images = images.to(device)

            logits, _ = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            for subject, probability, label in zip(
                subjects,
                probabilities.cpu(),
                labels,
            ):
                subject_probabilities[subject] = float(
                    probability.item()
                )
                subject_labels[subject] = int(label.item())

    return subject_probabilities, subject_labels


def calculate_metrics(
    subject_probabilities,
    subject_labels,
    threshold,
):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for subject, probability in subject_probabilities.items():
        prediction = 1 if probability >= threshold else 0
        label = subject_labels[subject]

        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_validation_threshold(
    subject_probabilities,
    subject_labels,
):
    """
    Select the threshold using validation balanced accuracy.

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
        "Balanced accuracy: "
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

    experiment_c_path = Path(EXPERIMENT_C_CHECKPOINT)
    experiment_l_path = Path(EXPERIMENT_L_CHECKPOINT)

    if not experiment_c_path.exists():
        raise FileNotFoundError(
            f"Experiment C checkpoint not found: "
            f"{experiment_c_path}"
        )

    if not experiment_l_path.exists():
        raise FileNotFoundError(
            f"Experiment L checkpoint not found: "
            f"{experiment_l_path}"
        )

    validation_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="val",
    )

    test_dataset = ADNISubjectDataset(
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

    model = ResNet18MeanFeaturePooling(
        checkpoint_path=EXPERIMENT_C_CHECKPOINT,
        freeze_feature_extractor=True,
    )

    model.load_state_dict(
        torch.load(
            EXPERIMENT_L_CHECKPOINT,
            map_location=device,
        )
    )

    model = model.to(device)

    (
        validation_probabilities,
        validation_labels,
    ) = collect_predictions(
        model,
        validation_loader,
        device,
    )

    (
        test_probabilities,
        test_labels,
    ) = collect_predictions(
        model,
        test_loader,
        device,
    )

    threshold, validation_metrics = (
        select_validation_threshold(
            validation_probabilities,
            validation_labels,
        )
    )

    test_metrics = calculate_metrics(
        test_probabilities,
        test_labels,
        threshold,
    )

    print("\n" + "=" * 72)
    print("Experiment L — Mean Feature Aggregation")
    print("=" * 72)

    print_metrics(
        "Validation metrics",
        threshold,
        validation_metrics,
    )

    print_metrics(
        "Test metrics using fixed validation threshold",
        threshold,
        test_metrics,
    )


if __name__ == "__main__":
    main()