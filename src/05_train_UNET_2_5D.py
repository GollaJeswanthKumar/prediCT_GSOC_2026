"""
05_train.py (2.5D U-Net)

Train 2.5D U-Net using stacked slices.

- Saves metrics to CSV
- Saves plots
- Saves best model
"""

import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from config import EPOCHS, LR
from dataset import get_loaders


K = 3  # number of slices (3 or 5)
LOG_DIR = "UNET_2_5D_logs"
MODEL_PATH = os.path.join(LOG_DIR, "best_model.pth")


## Model - UNET_2.5D 

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=K, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        in_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bridge = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bridge(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)


## Loss & Metric 

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth)).item()


## Training 
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMPORTANT: pass K to dataset
    train_loader, val_loader, _, _ = get_loaders(k=K)

    model = UNet2D(in_channels=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # ---- Logging ----
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "training_log.csv")

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])

    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                val_loss += combined_loss(preds, masks).item()

                for i in range(imgs.size(0)):
                    val_dice += dice_score(preds[i], masks[i])

        val_loss /= len(val_loader)
        val_dice /= len(val_loader.dataset)

        # ---- Save metrics ----
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_dice])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("  → Saved best model")

    # ---- Plot ----
    df = pd.read_csv(log_file)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(os.path.join(LOG_DIR, "loss_curve.png"))

    plt.figure()
    plt.plot(df["epoch"], df["val_dice"], label="Dice")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Curve")
    plt.savefig(os.path.join(LOG_DIR, "dice_curve.png"))

    print("\nTraining finished")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Logs saved in: {LOG_DIR}")

    return model, history


if __name__ == "__main__":
    train()