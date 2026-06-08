import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = Path("/Users/krystalmiao/Downloads/ADNI1_Complete_1Yr_1.5T_6_06_2026.csv")

OUTPUT_PATH = PROJECT_ROOT / "data" / "filtered_metadata.csv"

df = pd.read_csv(CSV_PATH)

filtered = df[

    df["Group"].isin(["AD", "CN"])

    & (df["Visit"] == "sc")

    & df["Description"].str.contains("MPR", na=False)

    & ~df["Description"].str.contains("MPR-R", na=False)

].copy()

print("Filtered group counts:")

print(filtered["Group"].value_counts())

print("Total:", len(filtered))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

filtered.to_csv(OUTPUT_PATH, index=False)

print(f"Saved to: {OUTPUT_PATH}")