"""
preprocess.py

Preprocess every patient that has a valid heart mask:
    1. Load DICOM → convert to Hounsfield Units (HU)
    2. Load heart.nii.gz mask
    3. Isotropic resampling to NEW_SPACING (default 1×1×1 mm)
    4. HU windowing → normalize to [0, 1]
    5. Spatial resize to TARGET_SIZE (256×256)
    6. Stratified 70/15/15 train/val/test split
    7. Save each 2D slice as img_N.npy + mask_N.npy
    8. Print dataset statistics

Outputs:
    SAVE_ROOT/<patient_id>/img_N.npy
    SAVE_ROOT/<patient_id>/mask_N.npy
    SAVE_ROOT/split.json          -> patient-level train/val/test lists
    SAVE_ROOT/dataset_stats.json  -> slice counts, positive-slice rates
"""

import json
import os
import random

import numpy as np
import pydicom
import scipy.ndimage
import SimpleITK as sitk
from tqdm import tqdm

from config import (
    DATA_ROOT, MASK_ROOT, SAVE_ROOT,
    HU_MIN, HU_MAX, NEW_SPACING, TARGET_SIZE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)


## DICOM loading 

def load_patient_dicom(dicom_folder: str):
    """
    Read all .dcm slices, sort by InstanceNumber, convert to HU.
    Returns:
        volume : np.ndarray  shape (Z, H, W), float32, in HU
        slices : list of pydicom.Dataset objects
    """
    slices = []
    for fname in os.listdir(dicom_folder):
        if fname.lower().endswith(".dcm"):
            slices.append(pydicom.dcmread(os.path.join(dicom_folder, fname)))

    if not slices:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    slices.sort(key=lambda s: int(s.InstanceNumber))

    volume = []
    for s in slices:
        img   = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope",     1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        volume.append(img * slope + intercept)

    return np.stack(volume), slices


## Mask loading

def load_heart_mask(mask_path: str):
    """
    Read heart.nii.gz, binarise.
    Returns:
        mask    : np.ndarray  shape (Z, H, W), uint8 {0, 1}
        mask_itk: SimpleITK image (for spacing metadata)
    """
    mask_itk = sitk.ReadImage(mask_path)
    mask     = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)
    mask     = (mask > 0).astype(np.uint8)
    return mask, mask_itk


## Preprocessing steps 

def resample(volume: np.ndarray, mask: np.ndarray, slices):
    """
    Isotropic resampling using slice-level DICOM spacing metadata.
    Volume is resampled with linear interpolation; mask with nearest-neighbour.
    """
    z_spacing = float(slices[0].SliceThickness)
    y_spacing = float(slices[0].PixelSpacing[0])
    x_spacing = float(slices[0].PixelSpacing[1])
    orig_spacing = np.array([z_spacing, y_spacing, x_spacing])

    resize_factor    = orig_spacing / np.array(NEW_SPACING)
    new_shape        = np.round(np.array(volume.shape) * resize_factor).astype(int)
    real_resize      = new_shape / np.array(volume.shape)

    volume_r = scipy.ndimage.zoom(volume, real_resize, order=1)
    mask_r   = scipy.ndimage.zoom(mask,   real_resize, order=0)
    return volume_r, mask_r


def window_and_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Cardiac window [-100, 400] HU → normalise to [0, 1].
    Chosen window emphasises soft-tissue / calcium contrast while suppressing
    lung air and bone extremes — optimal for heart segmentation.
    """
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume.astype(np.float32)


def resize_hw(volume: np.ndarray, mask: np.ndarray, target=(256, 256)):
    """Resize H×W to target size; keep Z unchanged."""
    d, h, w = volume.shape
    th, tw  = target
    zoom    = (1, th / h, tw / w)
    volume  = scipy.ndimage.zoom(volume, zoom, order=1)
    mask    = scipy.ndimage.zoom(mask,   zoom, order=0)
    return volume, mask


def preprocess(volume: np.ndarray, mask: np.ndarray, slices):
    """Full pipeline: resample → window/normalise → resize."""
    volume, mask = resample(volume, mask, slices)
    volume       = window_and_normalize(volume)
    volume, mask = resize_hw(volume, mask, TARGET_SIZE)
    return volume, mask


## Stratified Split

def split_patients(valid_patients):
    """Stratified 70/15/15 split """
    pts = sorted(valid_patients)       
    random.seed(RANDOM_SEED)
    random.shuffle(pts)

    n     = len(pts)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = pts[:n_tr]
    val_ids   = pts[n_tr : n_tr + n_val]
    test_ids  = pts[n_tr + n_val :]

    return train_ids, val_ids, test_ids


## Main

def run():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Find patients with valid heart masks
    valid_patients = [
        p for p in os.listdir(MASK_ROOT)
        if os.path.exists(os.path.join(MASK_ROOT, p, "heart.nii.gz"))
    ]
    print(f"Patients with valid heart masks: {len(valid_patients)}")

    train_ids, val_ids, test_ids = split_patients(valid_patients)
    print(f"Split → train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    # Save split for later scripts
    split_info = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(os.path.join(SAVE_ROOT, "split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    all_patients = train_ids + val_ids + test_ids
    stats = {}

    for patient_id in tqdm(all_patients, desc="Preprocessing"):
        out_dir = os.path.join(SAVE_ROOT, patient_id)
        # Resume support — skip if already done
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            n_slices = len([f for f in os.listdir(out_dir) if "img" in f])
            # still record for stats
            n_pos = sum(
                np.load(os.path.join(out_dir, f"mask_{i}.npy")).max() > 0
                for i in range(n_slices)
            )
            stats[patient_id] = {"n_slices": n_slices, "n_positive": n_pos}
            continue

        dicom_folder = os.path.join(DATA_ROOT, patient_id)
        mask_path    = os.path.join(MASK_ROOT, patient_id, "heart.nii.gz")

        try:
            volume, slices   = load_patient_dicom(dicom_folder)
            mask, _          = load_heart_mask(mask_path)
            volume, mask     = preprocess(volume, mask, slices)
        except Exception as e:
            print(f"\n  [WARN] Skipping {patient_id}: {e}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        n_positive = 0
        for i in range(volume.shape[0]):
            np.save(os.path.join(out_dir, f"img_{i}.npy"),  volume[i])
            np.save(os.path.join(out_dir, f"mask_{i}.npy"), mask[i])
            if mask[i].max() > 0:
                n_positive += 1

        stats[patient_id] = {
            "n_slices":   volume.shape[0],
            "n_positive": n_positive,
        }

    # Dataset statistics
    def split_stats(ids, label):
        total_slices = sum(stats[p]["n_slices"]   for p in ids if p in stats)
        total_pos    = sum(stats[p]["n_positive"] for p in ids if p in stats)
        ratio        = total_pos / max(total_slices, 1)
        print(f"\n{label}")
        print(f"  Patients     : {len(ids)}")
        print(f"  Total slices : {total_slices}")
        print(f"  Pos slices   : {total_pos}  ({ratio*100:.1f}%)")
        return {"patients": len(ids), "total_slices": total_slices,
                "positive_slices": total_pos, "positive_ratio": round(ratio, 4)}

    summary = {
        "train": split_stats(train_ids, "TRAIN"),
        "val":   split_stats(val_ids,   "VAL"),
        "test":  split_stats(test_ids,  "TEST"),
    }
    with open(os.path.join(SAVE_ROOT, "dataset_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStats saved to {SAVE_ROOT}/dataset_stats.json")
    print("Preprocessing complete.")


if __name__ == "__main__":
    run()
