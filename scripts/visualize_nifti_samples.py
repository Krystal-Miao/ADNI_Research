from pathlib import Path

import random

import nibabel as nib

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "raw_data_one_scan"

OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nii_files = list(DATA_ROOT.rglob("*.nii"))

random.seed(42)

sample_files = random.sample(nii_files, 6)

for i, p in enumerate(sample_files, start=1):

    img = nib.load(str(p))

    data = img.get_fdata()

    z = data.shape[2] // 2

    slice_img = data[:, :, z]

    plt.figure(figsize=(5, 5))

    plt.imshow(slice_img.T, cmap="gray", origin="lower")

    plt.axis("off")

    plt.title(f"{p.parent.name} | shape={data.shape}")

    out_path = OUTPUT_DIR / f"sample_{i}.png"

    plt.savefig(out_path, bbox_inches="tight", dpi=150)

    plt.close()

    print(f"Saved: {out_path}")