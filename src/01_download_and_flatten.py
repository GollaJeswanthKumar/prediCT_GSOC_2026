import os
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from config import DATA_ROOT

# Downloading the dataset by following the PrediCT github instructions  

AZURE_SAS_URL = (
    "https://aimistanforddatasets01.blob.core.windows.net/"
    "cocacoronarycalciumandchestcts-2?"
    "sv=2019-02-02&sr=c&sig=3xYLlaEifI%2BHwTZwttd%2BaHYvGidisLKQHVu0V1rpg%2Fk%3D"
    "&st=2026-03-24T11%3A32%3A34Z&se=2026-04-23T11%3A37%3A34Z&sp=rl"
)

DOWNLOAD_DEST = str(Path(DATA_ROOT).parents[2])


def download_dataset():
    
    print("Downloading COCA dataset via azcopy …")
    print(f"  Destination : {DOWNLOAD_DEST}")

    os.makedirs(DOWNLOAD_DEST, exist_ok=True)

    result = subprocess.run(
        ["azcopy", "copy", AZURE_SAS_URL, DOWNLOAD_DEST,
         "--recursive", "--from-to", "BlobLocal"],
        check=True,
    )
    print("Download complete.")


## Flattening the DICOM files from nested sub-directories to the patient root

def flatten_dicom_folders():

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"Patient directory not found: {DATA_ROOT}\n"
            "Run the download step first, or fix DATA_ROOT in config.py"
        )

    patients = sorted(os.listdir(DATA_ROOT))
    print(f"\nFlattening {len(patients)} patient folders …")

    for patient_id in tqdm(patients, desc="Flattening"):
        patient_dir = Path(DATA_ROOT) / patient_id

        # Collect all .dcm files in any sub-directory
        nested_dcms = [
            p for p in patient_dir.rglob("*.dcm")
            if p.parent != patient_dir      # already at root → skip
        ]

        for dcm_path in nested_dcms:
            target = patient_dir / dcm_path.name
            if target.exists():
                # avoid collision by prepending parent folder name
                target = patient_dir / f"{dcm_path.parent.name}_{dcm_path.name}"
            shutil.move(str(dcm_path), str(target))

        # Remove now-empty sub-directories
        for subfolder in sorted(patient_dir.iterdir(), reverse=True):
            if subfolder.is_dir():
                try:
                    shutil.rmtree(subfolder)
                except OSError:
                    pass

    print("Flatten complete.")


# Final check to ensure we have .dcm files at the patient root level

def sanity_check():
    patients = sorted(os.listdir(DATA_ROOT))
    sample   = patients[0]
    dcms     = list((Path(DATA_ROOT) / sample).glob("*.dcm"))
    print(f"\nSanity check — patient '{sample}': {len(dcms)} .dcm slices")
    assert len(dcms) > 0, "No DICOM files found after flattening!"
    print("OK — dataset looks ready.\n")


if __name__ == "__main__":
    download_dataset()
    flatten_dicom_folders()
    sanity_check()
