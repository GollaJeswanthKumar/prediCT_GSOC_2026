"""
sanity_check.py

Quick checks for 2.5D pipeline:
- Data loading
- Shapes
- Model forward pass
"""

import torch
from dataset import get_loaders
from train import UNet2D

K = 3 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ##Load data 
    train_loader, _, _, _ = get_loaders(k=K)

    imgs, masks = next(iter(train_loader))

    print("Data check:")
    print(f"  Input shape : {imgs.shape}")   # expected (B, K, H, W)
    print(f"  Mask shape  : {masks.shape}")  # expected (B, 1, H, W)

    ## Model check
    model = UNet2D(in_channels=K).to(device)

    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)

    print("\nModel check:")
    print(f"  Output shape: {outputs.shape}")  # expected (B, 1, H, W)

    ## Basic value checks
    print("\nValue check:")
    print(f"  Input min/max : {imgs.min().item():.3f} / {imgs.max().item():.3f}")
    print(f"  Output min/max: {outputs.min().item():.3f} / {outputs.max().item():.3f}")

    print("\n Sanity check passed — ready to train")


if __name__ == "__main__":
    main()