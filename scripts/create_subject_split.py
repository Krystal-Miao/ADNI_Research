import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

INPUT_CSV = Path("data/slice_metadata.csv")

OUTPUT_CSV = Path("data/slice_metadata_split.csv")

df = pd.read_csv(INPUT_CSV)

subjects = df[["subject", "label"]].drop_duplicates()

train_subjects, temp_subjects = train_test_split(

    subjects,

    test_size=0.30,

    random_state=42,

    stratify=subjects["label"],

)

val_subjects, test_subjects = train_test_split(

    temp_subjects,

    test_size=0.50,

    random_state=42,

    stratify=temp_subjects["label"],

)

split_map = {}

for s in train_subjects["subject"]:

    split_map[s] = "train"

for s in val_subjects["subject"]:

    split_map[s] = "val"

for s in test_subjects["subject"]:

    split_map[s] = "test"

df["split"] = df["subject"].map(split_map)

print("Subject counts:")

print(df[["subject", "label", "split"]].drop_duplicates()["split"].value_counts())

print("\nSubject label counts by split:")

print(df[["subject", "label", "split"]].drop_duplicates().groupby(["split", "label"]).size())

print("\nSlice counts by split:")

print(df["split"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved to {OUTPUT_CSV}")