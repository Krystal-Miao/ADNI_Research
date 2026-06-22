from torch.utils.data import DataLoader

from dataset import ADNISliceDataset

csv_path = "data/slice_metadata_split.csv"

train_dataset = ADNISliceDataset(csv_path, split="train")

val_dataset = ADNISliceDataset(csv_path, split="val")

test_dataset = ADNISliceDataset(csv_path, split="test")

print("Train slices:", len(train_dataset))

print("Val slices:", len(val_dataset))

print("Test slices:", len(test_dataset))

loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

images, labels, subjects = next(iter(loader))

print("Image batch shape:", images.shape)

print("Label batch shape:", labels.shape)

print("Subjects:", subjects[:3])