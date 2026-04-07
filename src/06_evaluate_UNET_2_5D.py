"""
evaluate.py 

Evaluate trained 2.5D U-Net:
- Dice score
- Inference speed
- Sample predictions
- Save results
"""

import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_loaders
from train import UNet2D, dice_score


K = 3
MODEL_PATH = "UNET_2_5D_logs/best_model.pth"
RESULTS_DIR = "UNET_2_5D_results"


## Load model

def load_model(path, device):
    model = UNet2D(in_channels=K)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


## Dice evaluation

def evaluate_test_set(model, test_loader, device):
    scores = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            for i in range(imgs.size(0)):
                scores.append(dice_score(preds[i], masks[i]))

    scores = np.array(scores)

    result = {
        "num_slices": int(len(scores)),
        "mean_dice": float(scores.mean()),
        "std_dice": float(scores.std()),
        "min_dice": float(scores.min()),
        "max_dice": float(scores.max()),
    }

    print("\nTest Dice Results")
    print(f"  Mean Dice : {result['mean_dice']:.4f}")

    return result


## Inference timing 

def benchmark_inference(model, test_loader, device, n_batches=10):
    times = []

    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            if i >= n_batches:
                break

            imgs = imgs.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(imgs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            times.append((t1 - t0) / imgs.size(0) * 1000)

    result = {
        "mean_ms_per_slice": float(np.mean(times)),
        "std_ms_per_slice": float(np.std(times)),
    }

    print(f"Inference: {result['mean_ms_per_slice']:.2f} ms/slice")

    return result


## Visualisation

def visualize_predictions(model, dataset, device, n=5, save_dir=None):
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()

        # center slice for display
        center_img = img[K // 2].numpy()

        mask_np = mask.squeeze().numpy()
        pred_bin = (pred > 0.5).astype(float)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))

        ax[0].imshow(center_img, cmap="gray")
        ax[0].set_title("Input (center)")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")

        ax[2].imshow(pred_bin, cmap="gray")
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        ax[3].imshow(center_img, cmap="gray")
        ax[3].imshow(pred_bin, cmap="jet", alpha=0.4)
        ax[3].set_title("Overlay")
        ax[3].axis("off")

        plt.tight_layout()

        path = os.path.join(save_dir, f"sample_{i+1}.png")
        plt.savefig(path)
        print(f"Saved: {path}")

        plt.close()


## Main 

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH, device)

    _, _, test_loader, test_dataset = get_loaders(k=K)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dice_results = evaluate_test_set(model, test_loader, device)
    timing_results = benchmark_inference(model, test_loader, device)

    vis_dir = os.path.join(RESULTS_DIR, "visuals")
    visualize_predictions(model, test_dataset, device, n=5, save_dir=vis_dir)

    results = {
        "dice": dice_results,
        "timing": timing_results,
    }

    save_path = os.path.join(RESULTS_DIR, "eval_results.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()