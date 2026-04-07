import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, NUM_WORKERS, SAVE_ROOT


## Dataset

class COCADataset(Dataset):
    """
    2.5D dataset:
    Input  : (K, H, W)  → stacked neighbouring slices
    Target : (1, H, W)  → center slice mask
    """

    def __init__(self, patient_data: dict, train: bool = True, k: int = 3):
        """
        Args:
            patient_data : dict {patient_id: [(img_path, mask_path), ...]}
            train        : apply augmentation
            k            : number of slices (odd - 3,5,7,..)
        """
        self.patient_data = patient_data
        self.train = train
        self.k = k
        self.half_k = k // 2

        # Flatten into (patient_id, slice_idx)
        self.samples = []
        for pid, slices in patient_data.items():
            for i in range(len(slices)):
                self.samples.append((pid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, center_idx = self.samples[idx]
        slices = self.patient_data[pid]

        # ---- Build stack ----
        stack = []

        for offset in range(-self.half_k, self.half_k + 1):
            new_idx = center_idx + offset

            # Clamp to stay within same patient
            new_idx = max(0, min(len(slices) - 1, new_idx))

            img_path, _ = slices[new_idx]
            img = np.load(img_path).astype(np.float32)
            stack.append(img)

        stack = np.stack(stack, axis=0)  # (K, H, W)

        # ---- Center mask ----
        _, mask_path = slices[center_idx]
        mask = np.load(mask_path).astype(np.float32)

        if self.train:
            stack, mask = self._augment(stack, mask)

        stack = torch.from_numpy(stack)          # (K, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return stack, mask

    ## Augmentation 
    @staticmethod
    def _augment(stack, mask):
        """Apply same transform across all slices"""

        # Horizontal flip
        if random.random() < 0.5:
            stack = np.flip(stack, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if random.random() < 0.5:
            stack = np.flip(stack, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Rotation
        if random.random() < 0.3:
            k = random.randint(1, 3)
            stack = np.rot90(stack, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()

        # Intensity jitter (only on image stack)
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-0.05, 0.05)
            stack = np.clip(stack * alpha + beta, 0.0, 1.0)

        return stack, mask


## Build patient-wise index

def build_patient_index(patient_ids, save_root=SAVE_ROOT):
    """
    Returns:
        dict {patient_id: [(img_path, mask_path), ...]}
    """
    patient_dict = {}

    for pid in patient_ids:
        path = os.path.join(save_root, pid)
        if not os.path.isdir(path):
            continue

        img_files = sorted(
            f for f in os.listdir(path) if f.startswith("img_") and f.endswith(".npy")
        )

        slices = []
        for fname in img_files:
            i = fname.split("_")[1].split(".")[0]

            img_path = os.path.join(path, f"img_{i}.npy")
            mask_path = os.path.join(path, f"mask_{i}.npy")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                slices.append((img_path, mask_path))

        if len(slices) > 0:
            patient_dict[pid] = slices

    return patient_dict


## DataLoaders

def get_loaders(split_json=None, k=3):
    if split_json is None:
        split_json = os.path.join(SAVE_ROOT, "split.json")

    with open(split_json) as f:
        split = json.load(f)

    train_data = build_patient_index(split["train"])
    val_data = build_patient_index(split["val"])
    test_data = build_patient_index(split["test"])

    train_ds = COCADataset(train_data, train=True, k=k)
    val_ds = COCADataset(val_data, train=False, k=k)
    test_ds = COCADataset(test_data, train=False, k=k)

    print(f"Samples → train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, test_ds