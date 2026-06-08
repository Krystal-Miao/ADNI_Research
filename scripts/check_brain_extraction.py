from pathlib import Path

import random

import nibabel as nib

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "raw_data_one_scan"

nii_files = list(DATA_ROOT.rglob("*.nii"))

random.seed(42)

sample_file = random.choice(nii_files)

print("Selected file:")

print(sample_file)

img = nib.load(str(sample_file))

data = img.get_fdata()

print("Shape:", data.shape)

# sagittal

x = data.shape[0] // 2

# coronal

y = data.shape[1] // 2

# axial

z = data.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(data[x, :, :].T, cmap="gray", origin="lower")

axes[0].set_title("Sagittal")

axes[1].imshow(data[:, y, :].T, cmap="gray", origin="lower")

axes[1].set_title("Coronal")

axes[2].imshow(data[:, :, z].T, cmap="gray", origin="lower")

axes[2].set_title("Axial")

for ax in axes:

    ax.axis("off")

plt.tight_layout()

plt.show()