import pandas as pd

CSV_PATH = "data/slice_metadata_split.csv"

df = pd.read_csv(CSV_PATH)

subject_df = df[["subject", "label", "split"]].drop_duplicates()

train_subjects = subject_df[subject_df["split"] == "train"]

counts = train_subjects["label"].value_counts().sort_index()

cn_count = counts[0]

ad_count = counts[1]

print("Train subject counts:")

print("CN:", cn_count)

print("AD:", ad_count)

print("\nRecommended class weights:")

print("CN weight:", 1.0)

print("AD weight:", cn_count / ad_count)