import os

from pathlib import Path

def get_config_dict():

    project_root = Path(__file__).resolve().parent

    config = {}

    config["data_path"] = project_root / "data" / "raw_data_test"

    config["re_process"] = False

    resolution_mm = 2

    config["reference_atlas_location"] = (

        Path(os.environ["FSLDIR"])

        / "data"

        / "standard"

        / f"MNI152_T1_{resolution_mm}mm_brain.nii.gz"

    )

    config["axial_size"] = 90

    config["save_2d"] = True

    return config