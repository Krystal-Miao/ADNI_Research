from collections import defaultdict

from pathlib import Path

import csv

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import models

from training.dataset import ADNISliceDataset

CSV_PATH = "data/slice_metadata_split.csv"

BATCH_SIZE = 16

OUTPUT_PATH = "reports/evaluation_A_to_I.csv"

EXPERIMENTS = {

    "A": {

        "name": "Pretrained frozen baseline",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_A_resnet18_baseline.pt"

        ),

    },

    "B": {

        "name": "Class weights",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_B_resnet18_class_weights.pt"

        ),

    },

    "C": {

        "name": "Layer4 fine-tuning",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_C_resnet18_layer4.pt"

        ),

    },

    "D": {

        "name": "Layer4 + class weights",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_D_resnet18_layer4_class_weights.pt"

        ),

    },

    "E": {

        "name": "Training from scratch",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_E_resnet18_scratch.pt"

        ),

    },

    "F": {

        "name": "Layer4 + weight decay",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_F_resnet18_layer4_weight_decay.pt"

        ),

    },

    "G": {

        "name": "Layer4 + early stopping",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_G_resnet18_layer4_early_stopping.pt"

        ),

    },

    "H": {

        "name": "Layer4 + weight decay + early stopping",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_H_resnet18_layer4_weight_decay_early_stopping.pt"

        ),

    },

    "I": {

        "name": "Layer4 + mild intensity scaling",

        "checkpoint": (

            "training/checkpoints/"

            "best_experiment_I_resnet18_layer4_intensity_augmentation.pt"

        ),

    },

}

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")

def create_model():

    """

    All experiments use the same ResNet18 architecture.

    Pretrained initialization is not required during evaluation because

    the complete saved state_dict is loaded afterward.

    """

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def get_subject_probabilities(model, loader, device):

    """

    Run slice-level prediction and average AD probabilities for each subject.

    """

    model.eval()

    subject_probabilities = defaultdict(list)

    subject_labels = {}

    with torch.no_grad():

        for images, labels, subjects in loader:

            images = images.to(device)

            outputs = model(images)

            ad_probabilities = torch.softmax(outputs, dim=1)[:, 1]

            for subject, probability, label in zip(

                subjects,

                ad_probabilities.cpu(),

                labels,

            ):

                subject_probabilities[subject].append(

                    float(probability.item())

                )

                subject_labels[subject] = int(label.item())

    averaged_probabilities = {}

    for subject, probabilities in subject_probabilities.items():

        averaged_probabilities[subject] = (

            sum(probabilities) / len(probabilities)

        )

    return averaged_probabilities, subject_labels

def calculate_metrics(subject_probabilities, subject_labels, threshold):

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

    accuracy = (tp + tn) / total if total > 0 else 0.0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

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

def select_validation_threshold(

    subject_probabilities,

    subject_labels,

):

    """

    Select the threshold using validation balanced accuracy.

    Tie-breaking order:

    1. Higher balanced accuracy

    2. Higher sensitivity

    3. Threshold closer to 0.50

    """

    thresholds = np.arange(0.05, 0.951, 0.01)

    best_threshold = None

    best_metrics = None

    best_score = None

    for threshold in thresholds:

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

def print_experiment_result(

    experiment_letter,

    experiment_name,

    threshold,

    validation_metrics,

    test_metrics,

):

    print("\n" + "=" * 80)

    print(f"Experiment {experiment_letter}: {experiment_name}")

    print("=" * 80)

    print(f"Selected validation threshold: {threshold:.2f}")

    print("\nValidation metrics:")

    print(

        f"Accuracy={validation_metrics['accuracy']:.4f} | "

        f"Sensitivity={validation_metrics['sensitivity']:.4f} | "

        f"Specificity={validation_metrics['specificity']:.4f} | "

        f"Balanced Acc={validation_metrics['balanced_accuracy']:.4f}"

    )

    print("\nTest metrics using fixed validation threshold:")

    print(

        f"Accuracy={test_metrics['accuracy']:.4f} | "

        f"Sensitivity={test_metrics['sensitivity']:.4f} | "

        f"Specificity={test_metrics['specificity']:.4f} | "

        f"Balanced Acc={test_metrics['balanced_accuracy']:.4f}"

    )

    print(

        "Confusion Matrix: "

        f"TP={test_metrics['tp']} "

        f"FP={test_metrics['fp']} "

        f"TN={test_metrics['tn']} "

        f"FN={test_metrics['fn']}"

    )

def save_results(results):

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [

        "experiment",

        "name",

        "validation_threshold",

        "validation_accuracy",

        "validation_sensitivity",

        "validation_specificity",

        "validation_balanced_accuracy",

        "test_accuracy",

        "test_sensitivity",

        "test_specificity",

        "test_balanced_accuracy",

        "test_tp",

        "test_fp",

        "test_tn",

        "test_fn",

    ]

    with output_path.open("w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(results)

    print(f"\nSaved consolidated results to: {output_path}")

def main():

    device = get_device()

    print("Device:", device)

    validation_dataset = ADNISliceDataset(

        CSV_PATH,

        split="val",

        augment=False,

        intensity_augment=False,

    )

    test_dataset = ADNISliceDataset(

        CSV_PATH,

        split="test",

        augment=False,

        intensity_augment=False,

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

    all_results = []

    for experiment_letter, config in EXPERIMENTS.items():

        checkpoint_path = Path(config["checkpoint"])

        if not checkpoint_path.exists():

            print(

                f"\nSkipping Experiment {experiment_letter}: "

                f"checkpoint not found at {checkpoint_path}"

            )

            continue

        model = create_model()

        state_dict = torch.load(

            checkpoint_path,

            map_location=device,

        )

        model.load_state_dict(state_dict)

        model = model.to(device)

        (

            validation_probabilities,

            validation_labels,

        ) = get_subject_probabilities(

            model,

            validation_loader,

            device,

        )

        threshold, validation_metrics = select_validation_threshold(

            validation_probabilities,

            validation_labels,

        )

        test_probabilities, test_labels = get_subject_probabilities(

            model,

            test_loader,

            device,

        )

        test_metrics = calculate_metrics(

            test_probabilities,

            test_labels,

            threshold,

        )

        print_experiment_result(

            experiment_letter,

            config["name"],

            threshold,

            validation_metrics,

            test_metrics,

        )

        all_results.append({

            "experiment": experiment_letter,

            "name": config["name"],

            "validation_threshold": threshold,

            "validation_accuracy": validation_metrics["accuracy"],

            "validation_sensitivity": validation_metrics["sensitivity"],

            "validation_specificity": validation_metrics["specificity"],

            "validation_balanced_accuracy": (

                validation_metrics["balanced_accuracy"]

            ),

            "test_accuracy": test_metrics["accuracy"],

            "test_sensitivity": test_metrics["sensitivity"],

            "test_specificity": test_metrics["specificity"],

            "test_balanced_accuracy": (

                test_metrics["balanced_accuracy"]

            ),

            "test_tp": test_metrics["tp"],

            "test_fp": test_metrics["fp"],

            "test_tn": test_metrics["tn"],

            "test_fn": test_metrics["fn"],

        })

    if not all_results:

        raise RuntimeError(

            "No checkpoints were evaluated. Check the checkpoint paths."

        )

    save_results(all_results)

if __name__ == "__main__":

    main()