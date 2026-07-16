from torch.utils.data import DataLoader

from training.subject_dataset import ADNISubjectDataset


CSV_PATH = "data/slice_metadata_split.csv"


def main():
    train_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="train",
    )

    val_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="val",
    )

    test_dataset = ADNISubjectDataset(
        CSV_PATH,
        split="test",
    )

    print("Train subjects:", len(train_dataset))
    print("Validation subjects:", len(val_dataset))
    print("Test subjects:", len(test_dataset))

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    images, labels, subjects = next(iter(loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Subjects:", subjects)

    assert images.shape == (2, 30, 3, 224, 224)
    assert labels.shape == (2,)

    print("Subject-level dataset test passed.")


if __name__ == "__main__":
    main()