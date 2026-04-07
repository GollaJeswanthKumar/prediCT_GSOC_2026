"""
evaluate.py

Evaluate trained U-Net on test set:
- Dice score
- Inference speed
- Sample predictions
- Save results to JSON
"""

import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL_SAVE_PATH, SAVE_ROOT
from dataset import get_loaders
from train import UNet2D, dice_score


## Load model 

def load_model(path, device):
    model = UNet2D()
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
    print(f"  Slices   : {result['num_slices']}")
    print(f"  Mean     : {result['mean_dice']:.4f}")
    print(f"  Std      : {result['std_dice']:.4f}")
    print(f"  Min/Max  : {result['min_dice']:.4f} / {result['max_dice']:.4f}")

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

            time_per_slice = (t1 - t0) / imgs.size(0) * 1000
            times.append(time_per_slice)

    result = {
        "mean_ms_per_slice": float(np.mean(times)),
        "std_ms_per_slice": float(np.std(times)),
    }

    print("\nInference Speed")
    print(f"  {result['mean_ms_per_slice']:.2f} ms/slice")

    return result


## Visualisation 

def visualize_predictions(model, dataset, device, n=5, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()

        img_np = img.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        pred_bin = (pred > 0.5).astype(float)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))

        ax[0].imshow(img_np, cmap="gray")
        ax[0].set_title("Input")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")

        ax[2].imshow(pred_bin, cmap="gray")
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        ax[3].imshow(img_np, cmap="gray")
        ax[3].imshow(pred_bin, cmap="jet", alpha=0.4)
        ax[3].set_title("Overlay")
        ax[3].axis("off")

        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, f"sample_{i+1}.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        else:
            plt.show()

        plt.close()


## Mainloop

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(MODEL_SAVE_PATH, device)

    _, _, test_loader, test_dataset = get_loaders()

    vis_dir = os.path.join(SAVE_ROOT, "visuals")

    dice_results = evaluate_test_set(model, test_loader, device)
    timing_results = benchmark_inference(model, test_loader, device)

    visualize_predictions(model, test_dataset, device, n=5, save_dir=vis_dir)

    results = {
        "dice": dice_results,
        "timing": timing_results,
    }

    os.makedirs(SAVE_ROOT, exist_ok=True)
    save_path = os.path.join(SAVE_ROOT, "eval_results.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    run()