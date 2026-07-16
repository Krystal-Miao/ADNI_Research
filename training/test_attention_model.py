import torch
from torch.utils.data import DataLoader

from training.attention_model import ResNet18AttentionPooling
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

    model = ResNet18AttentionPooling(
        checkpoint_path=CHECKPOINT_PATH,
        attention_hidden_size=128,
        freeze_feature_extractor=True,
    )

    logits, attention_weights = model(images)

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Subjects:", subjects)
    print("Logits:", logits.shape)
    print("Attention weights:", attention_weights.shape)
    print(
        "Attention sums:",
        attention_weights.sum(dim=1),
    )

    assert logits.shape == (2, 2)
    assert attention_weights.shape == (2, 30)

    assert torch.allclose(
        attention_weights.sum(dim=1),
        torch.ones(2),
        atol=1e-5,
    )

    print("Attention model test passed.")


if __name__ == "__main__":
    main()