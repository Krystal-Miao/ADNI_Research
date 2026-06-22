import pandas as pd

from pathlib import Path

# Files

METADATA_CSV = Path("data/filtered_metadata_one_scan.csv")

PREPROCESSED_DIR = Path("data/preprocessed_one_scan")

OUTPUT_CSV = Path("data/slice_metadata.csv")

# Load subject labels

metadata = pd.read_csv(METADATA_CSV)

# AD = 1, CN = 0

label_map = {

    "CN": 0,

    "AD": 1

}

subject_labels = {

    row["Subject"]: label_map[row["Group"]]

    for _, row in metadata.iterrows()

}

records = []

# Find all TIFF files

def get_slice_idx(path):

    name = path.stem

    return int(name.split("_slice")[-1])

for tiff_file in sorted(PREPROCESSED_DIR.rglob("*.tiff"), key=get_slice_idx):

    subject_id = tiff_file.parts[2]

    image_id = tiff_file.parts[-2]

    label = subject_labels.get(subject_id)

    if label is None:

        continue

    records.append({

        "subject": subject_id,

        "image_id": image_id,

        "label": label,

        "slice_path": str(tiff_file)

    })

df = pd.DataFrame(records)

print("Total slices:", len(df))

print(df["label"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to {OUTPUT_CSV}")