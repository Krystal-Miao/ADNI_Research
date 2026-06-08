import pandas as pd

from pathlib import Path

import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data" / "filtered_metadata_one_scan.csv"

IMAGE_ROOT = Path("/Users/krystalmiao/Downloads/ADNI")

OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw_data_one_scan"

df = pd.read_csv(METADATA_PATH)

nii_files = list(IMAGE_ROOT.rglob("*.nii"))

id_to_path = {}

for p in nii_files:

    image_id = p.parent.name  # e.g., I66462

    id_to_path[image_id] = p

missing = []

copied = 0

for _, row in df.iterrows():

    image_id = row["Image Data ID"]

    src = id_to_path.get(image_id)

    if src is None:

        missing.append(image_id)

        continue

    rel_path = src.relative_to(IMAGE_ROOT)

    dst = OUTPUT_ROOT / rel_path

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)

    copied += 1

print("Expected:", len(df))

print("Copied:", copied)

print("Missing:", len(missing))

if missing:

    print("First missing IDs:", missing[:10])

print(f"Saved one-scan-per-subject NIfTI files to: {OUTPUT_ROOT}")