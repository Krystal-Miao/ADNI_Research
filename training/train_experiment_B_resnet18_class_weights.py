import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import models

from collections import defaultdict

from dataset import ADNISliceDataset

from utils import get_class_weights

CSV_PATH = "data/slice_metadata_split.csv"

BATCH_SIZE = 16

EPOCHS = 5

LR = 1e-4

MODEL_SAVE_PATH = "training/best_experiment_B_resnet18_class_weights.pt"

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

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

            probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            for subject, prob, label in zip(subjects, probs.cpu(), labels.cpu()):

                subject_probs[subject].append(prob.item())

                subject_labels[subject] = int(label.item())

    slice_acc = correct / total

    subject_correct = 0

    for subject, probs in subject_probs.items():

        avg_prob = sum(probs) / len(probs)

        pred = 1 if avg_prob >= 0.5 else 0

        if pred == subject_labels[subject]:

            subject_correct += 1

    subject_acc = subject_correct / len(subject_probs)

    return total_loss / total, slice_acc, subject_acc

def main():

    device = get_device()

    print("Device:", device)

    train_dataset = ADNISliceDataset(CSV_PATH, split="train")

    val_dataset = ADNISliceDataset(CSV_PATH, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():

        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    class_weights = get_class_weights(CSV_PATH, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_val_subject_acc = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0

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

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total

        train_acc = correct / total

        val_loss, val_slice_acc, val_subject_acc = evaluate(

            model, val_loader, criterion, device

        )

        print(

            f"Epoch {epoch}/{EPOCHS} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Slice Acc: {train_acc:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Slice Acc: {val_slice_acc:.4f} | "

            f"Val Subject Acc: {val_subject_acc:.4f}"

        )

        if val_subject_acc > best_val_subject_acc:

            best_val_subject_acc = val_subject_acc

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print("Saved best model.")

    print("Training finished.")

    print("Best validation subject accuracy:", best_val_subject_acc)

if __name__ == "__main__":

    main()