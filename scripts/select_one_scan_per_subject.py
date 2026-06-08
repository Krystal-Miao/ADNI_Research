import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

metadata_path = PROJECT_ROOT / "data" / "filtered_metadata.csv"

output_path = PROJECT_ROOT / "data" / "filtered_metadata_one_scan.csv"

df = pd.read_csv(metadata_path)

df["image_num"] = df["Image Data ID"].str.replace("I", "", regex=False).astype(int)

one_scan = (

    df.sort_values(["Subject", "image_num"])

    .drop_duplicates(subset=["Subject"], keep="first")

    .drop(columns=["image_num"])

)

print("Original rows:", len(df))

print("Unique subjects:", df["Subject"].nunique())

print("After one-scan-per-subject:", len(one_scan))

print(one_scan["Group"].value_counts())

one_scan.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")