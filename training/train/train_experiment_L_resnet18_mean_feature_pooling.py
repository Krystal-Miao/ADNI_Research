from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.mean_feature_model import ResNet18MeanFeaturePooling
from training.subject_dataset import ADNISubjectDataset


CSV_PATH = "data/slice_metadata_split.csv"

EXPERIMENT_C_CHECKPOINT = (
    "training/checkpoints/"
    "best_experiment_C_resnet18_layer4.pt"
)

MODEL_SAVE_PATH = (
    "training/checkpoints/"
    "best_experiment_L_resnet18_mean_feature_pooling.pt"
)

BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
PATIENCE = 5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def calculate_metrics(predictions, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for prediction, label in zip(predictions, labels):
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

    balanced_accuracy = (sensitivity + specificity) / 2

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


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_subjects = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_subjects += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

    metrics = calculate_metrics(
        all_predictions,
        all_labels,
    )

    average_loss = total_loss / total_subjects

    return average_loss, metrics


def main():
    device = get_device()
    print("Device:", device)

    checkpoint_path = Path(EXPERIMENT_C_CHECKPOINT)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Experiment C checkpoint not found: {checkpoint_path}"
        )

    train_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="train",
    )

    validation_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ResNet18MeanFeaturePooling(
        checkpoint_path=EXPERIMENT_C_CHECKPOINT,
        freeze_feature_extractor=True,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Only the new subject-level classifier is trainable.
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LR,
    )

    best_validation_balanced_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_subjects = 0

        all_predictions = []
        all_labels = []

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_subjects += labels.size(0)

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )
            all_labels.extend(
                labels.detach().cpu().tolist()
            )

        train_loss = total_loss / total_subjects

        train_metrics = calculate_metrics(
            all_predictions,
            all_labels,
        )

        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train Balanced Acc: "
            f"{train_metrics['balanced_accuracy']:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_metrics['accuracy']:.4f} | "
            f"Val Sensitivity: "
            f"{validation_metrics['sensitivity']:.4f} | "
            f"Val Specificity: "
            f"{validation_metrics['specificity']:.4f} | "
            f"Val Balanced Acc: "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )

        current_score = validation_metrics[
            "balanced_accuracy"
        ]

        if current_score > best_validation_balanced_accuracy:
            best_validation_balanced_accuracy = current_score
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH,
            )

            print("Saved best model.")
        else:
            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}."
            )
            break

    print("Training finished.")
    print(
        "Best validation balanced accuracy:",
        best_validation_balanced_accuracy,
    )


if __name__ == "__main__":
    main()