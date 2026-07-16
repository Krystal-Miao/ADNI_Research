from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from training.dataset import ADNISliceDataset


CSV_PATH = "data/slice_metadata_split.csv"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-5

MODEL_SAVE_PATH = (
    "training/checkpoints/"
    "best_experiment_I_resnet18_layer4_intensity_augmentation.pt"
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    subject_probs = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():
        for images, labels, subjects in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for subject, probability, label in zip(
                subjects,
                probabilities.cpu(),
                labels.cpu(),
            ):
                subject_probs[subject].append(probability.item())
                subject_labels[subject] = int(label.item())

    slice_accuracy = correct / total

    subject_correct = 0

    predicted_AD = 0

    for subject, probs in subject_probs.items():

        avg_prob = sum(probs) / len(probs)

        pred = 1 if avg_prob >= 0.5 else 0

        print(

            f"{subject:15s}  "

            f"AvgProb={avg_prob:.3f}  "

            f"GT={subject_labels[subject]}  "

            f"Pred={pred}"

        )

        if pred == 1:

            predicted_AD += 1

        if pred == subject_labels[subject]:

            subject_correct += 1

    print(f"\nPredicted AD subjects: {predicted_AD}/{len(subject_probs)}")

    subject_acc = subject_correct / len(subject_probs)

    return total_loss / total, slice_accuracy, subject_acc


def main():
    device = get_device()
    print("Device:", device)

    # Apply intensity augmentation only to training images.
    train_dataset = ADNISliceDataset(
        CSV_PATH,
        split="train",
        augment=False,
        intensity_augment=True,
    )

    # Validation images must remain unchanged.
    val_dataset = ADNISliceDataset(
        CSV_PATH,
        split="val",
        augment=False,
        intensity_augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze the complete pretrained backbone first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune Layer4 only.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # No class weights in this experiment.
    criterion = nn.CrossEntropyLoss()

    # No weight decay: intensity augmentation is the only new variable.
    optimizer = torch.optim.Adam(
        list(model.layer4.parameters())
        + list(model.fc.parameters()),
        lr=LR,
    )

    best_val_subject_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Keep frozen backbone modules in evaluation mode so their
        # BatchNorm running statistics are not updated.

        model.conv1.eval()
        model.bn1.eval()
        model.relu.eval()
        model.maxpool.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

        # Layer4 and FC remain trainable.
        model.layer4.train()
        model.fc.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_slice_accuracy = correct / total

        (
            val_loss,
            val_slice_accuracy,
            val_subject_accuracy,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Slice Acc: {train_slice_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Slice Acc: {val_slice_accuracy:.4f} | "
            f"Val Subject Acc: {val_subject_accuracy:.4f}"
        )

        if val_subject_accuracy > best_val_subject_accuracy:
            best_val_subject_accuracy = val_subject_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model.")

    print("Training finished.")
    print(
        "Best validation subject accuracy:",
        best_val_subject_accuracy,
    )


if __name__ == "__main__":
    main()