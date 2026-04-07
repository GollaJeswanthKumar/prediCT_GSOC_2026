import os
import random

from tqdm import tqdm

from config import DATA_ROOT, MASK_ROOT, TARGET_MASK_COUNT, TOTALSEG_LICENSE, RANDOM_SEED

# Set license before importing TotalSegmentator
os.environ["TOTALSEG_LICENSE"] = TOTALSEG_LICENSE


def generate_heart_mask(dicom_folder: str, output_dir: str):
    """Call TotalSegmentator for a single patient. Saves heart.nii.gz to output_dir."""
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(
        dicom_folder,
        output_dir,
        task="total",
        fast=True,         # fast mode — good enough for coarse heart mask
    )


def run():
    os.makedirs(MASK_ROOT, exist_ok=True)

    all_patients = os.listdir(DATA_ROOT)
    random.seed(RANDOM_SEED)
    random.shuffle(all_patients)

    count = 0
    skipped = 0
    failed  = []

    print(f"Generating heart masks for up to {TARGET_MASK_COUNT} patients …")
    print(f"  DICOM root  : {DATA_ROOT}")
    print(f"  Mask root   : {MASK_ROOT}\n")

    for patient_id in tqdm(all_patients, desc="Patients"):
        if count >= TARGET_MASK_COUNT:
            break

        dicom_folder = os.path.join(DATA_ROOT, patient_id)
        output_dir   = os.path.join(MASK_ROOT, patient_id)
        heart_file   = os.path.join(output_dir, "heart.nii.gz")

        # Resume support — skip if already done
        if os.path.exists(heart_file):
            skipped += 1
            count   += 1
            continue

        os.makedirs(output_dir, exist_ok=True)

        try:
            generate_heart_mask(dicom_folder, output_dir)
            count += 1
        except Exception as e:
            print(f"\n  [WARN] Failed on {patient_id}: {e}")
            failed.append(patient_id)

    print(f"\nDone. Processed: {count}  |  Skipped (existing): {skipped}  |  Failed: {len(failed)}")
    if failed:
        print("Failed patients:", failed)

    # Show valid masks for downstream use
    valid = [
        p for p in os.listdir(MASK_ROOT)
        if os.path.exists(os.path.join(MASK_ROOT, p, "heart.nii.gz"))
    ]
    print(f"Valid patients with heart mask: {len(valid)}")


if __name__ == "__main__":
    run()
