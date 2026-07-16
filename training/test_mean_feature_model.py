import torch
from torch.utils.data import DataLoader

from training.mean_feature_model import ResNet18MeanFeaturePooling
from training.subject_dataset import ADNISubjectDataset


CSV_PATH = "data/slice_metadata_split.csv"

CHECKPOINT_PATH = (
    "training/checkpoints/"
    "best_experiment_C_resnet18_layer4.pt"
)


def main():
    dataset = ADNISubjectDataset(
        CSV_PATH,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, labels, subjects = next(iter(loader))

    model = ResNet18MeanFeaturePooling(
        checkpoint_path=CHECKPOINT_PATH,
        freeze_feature_extractor=True,
    )

    logits, subject_features = model(images)

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Subjects:", subjects)
    print("Logits:", logits.shape)
    print("Subject features:", subject_features.shape)

    assert images.shape == (2, 30, 3, 224, 224)
    assert logits.shape == (2, 2)
    assert subject_features.shape == (2, 512)

    print("Mean feature model test passed.")


if __name__ == "__main__":
    main()