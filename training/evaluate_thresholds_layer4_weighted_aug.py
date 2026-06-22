from collections import defaultdict

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import models

from dataset import ADNISliceDataset

CSV_PATH = "data/slice_metadata_split.csv"

MODEL_PATH = "training/best_resnet18_layer4_weighted_aug.pt"

BATCH_SIZE = 16

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")

def main():

    device = get_device()

    print("Device:", device)

    test_dataset = ADNISliceDataset(CSV_PATH, split="test")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model = model.to(device)

    model.eval()

    subject_probs = defaultdict(list)

    subject_labels = {}

    with torch.no_grad():

        for images, labels, subjects in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            for subject, prob, label in zip(subjects, probs.cpu(), labels.cpu()):

                subject_probs[subject].append(prob.item())

                subject_labels[subject] = int(label.item())

    subject_avg_probs = {

        subject: sum(probs) / len(probs)

        for subject, probs in subject_probs.items()

    }

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("\nThreshold Results:")

    print("Threshold | Acc   | Sensitivity | Specificity | TP FP TN FN")

    for threshold in thresholds:

        y_true = []

        y_pred = []

        for subject, avg_prob in subject_avg_probs.items():

            pred = 1 if avg_prob >= threshold else 0

            true = subject_labels[subject]

            y_true.append(true)

            y_pred.append(pred)

        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)

        acc = (tp + tn) / len(y_true)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(

            f"{threshold:>9.2f} | "

            f"{acc:.3f} | "

            f"{sensitivity:.3f}       | "

            f"{specificity:.3f}       | "

            f"{tp:>2} {fp:>2} {tn:>2} {fn:>2}"

        )

if __name__ == "__main__":

    main()