from pathlib import Path

import nibabel as nib

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "raw_data_one_scan"

nii_files = list(DATA_ROOT.rglob("*.nii"))

records = []

bad_files = []

for p in nii_files:

    try:

        img = nib.load(str(p))

        records.append({

            "path": str(p),

            "subject": p.relative_to(DATA_ROOT).parts[0],

            "image_id": p.parent.name,

            "shape": img.shape,

            "dtype": str(img.get_data_dtype())

        })

    except Exception as e:

        bad_files.append((str(p), str(e)))

df = pd.DataFrame(records)

print("Total NIfTI files:", len(nii_files))

print("Readable files:", len(df))

print("Bad files:", len(bad_files))

print("\nShape counts:")

print(df["shape"].value_counts())

print("\nDtype counts:")

print(df["dtype"].value_counts())

if bad_files:

    print("\nBad files:")

    for f, e in bad_files[:10]:

        print(f, e)

df.to_csv(PROJECT_ROOT / "data" / "nifti_check_summary.csv", index=False)

print("\nSaved summary to data/nifti_check_summary.csv")