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
    2D slice dataset.

    Each item:
        img  : float32 tensor  (1, H, W)  — normalised to [0, 1]
        mask : float32 tensor  (1, H, W)  — binary {0, 1}
    """

    def __init__(self, index: list, train: bool = True):
        """
        Args:
            index : list of (img_path, mask_path) tuples
            train : if True, apply augmentation
        """
        self.index = index
        self.train = train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, mask_path = self.index[idx]

        img  = np.load(img_path).astype(np.float32)    # (H, W)
        mask = np.load(mask_path).astype(np.float32)   # (H, W)

        if self.train:
            img, mask = self._augment(img, mask)

        img  = torch.from_numpy(img).unsqueeze(0)    # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        return img, mask

    ## Augmentation

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray):
        """Apply random flips, rotation, and intensity jitter."""

        # Horizontal flip
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if random.random() < 0.5:
            img  = np.flip(img,  axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90° rotation (k ∈ {1, 2, 3})
        if random.random() < 0.3:
            k    = random.randint(1, 3)
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()

        # Brightness + contrast jitter (image only)
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)   # contrast
            beta  = random.uniform(-0.05, 0.05)  # brightness
            img   = np.clip(img * alpha + beta, 0.0, 1.0)

        return img, mask


## Index builder

def build_index(patient_ids: list, save_root: str = SAVE_ROOT) -> list:
    """
    Walk each patient folder and collect (img_path, mask_path) pairs.
    Skips slices where the .npy file is missing on either side.
    """
    index = []
    for pid in patient_ids:
        path = os.path.join(save_root, pid)
        if not os.path.isdir(path):
            continue

        img_files = sorted(
            f for f in os.listdir(path) if f.startswith("img_") and f.endswith(".npy")
        )
        for fname in img_files:
            i         = fname.split("_")[1].split(".")[0]
            img_path  = os.path.join(path, f"img_{i}.npy")
            mask_path = os.path.join(path, f"mask_{i}.npy")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                index.append((img_path, mask_path))

    return index


## DataLoader

def get_loaders(split_json: str = None):
    """
    Build train / val / test DataLoaders from SAVE_ROOT/split.json.

    Returns:
        train_loader, val_loader, test_loader
    """
    if split_json is None:
        split_json = os.path.join(SAVE_ROOT, "split.json")

    with open(split_json) as f:
        split = json.load(f)

    train_index = build_index(split["train"])
    val_index   = build_index(split["val"])
    test_index  = build_index(split["test"])

    print(f"Index sizes → train: {len(train_index)}  val: {len(val_index)}  test: {len(test_index)}")

    train_ds = COCADataset(train_index, train=True)
    val_ds   = COCADataset(val_index,  train=False)
    test_ds  = COCADataset(test_index, train=False)

    common = dict(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    return train_loader, val_loader, test_loader, test_ds
