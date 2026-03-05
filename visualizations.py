# Import libraries: --->
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules: --->
from model import CrowdMLP
from dataset import CrowdDataset

# Import constants: --->
from const import DEVICE, DATASET_ROOT, BATCH_SIZE, VISUALISATIONS, MODEL, VISUAL_ROOT, IMAGE_COLOR, KERNEL, BOTTLENECK

# Load the test subset: --->
test_dataset = CrowdDataset(DATASET_ROOT, split = 'test')
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Load THE trained model: --->
model = CrowdMLP().to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location = DEVICE))
model.eval()

# Check whether the directory exists or not: --->
os.makedirs(VISUAL_ROOT, exist_ok=True)

with torch.no_grad():
    for inputs, gt_maps in test_loader:
        inputs = inputs.to(DEVICE)
        gt_maps = gt_maps.to(DEVICE)

        outputs = model(inputs)

        # Visualize a few samples: --->
        idx = 0
        while idx <  VISUALISATIONS:
            # image = inputs[idx].cpu().numpy().transpose(1, 2, 0)
            image = inputs[idx].cpu()

            print(f"\nImage {idx + 1} shape before transpose: {image.shape}")
            print(f"Min value in image {idx + 1}: {image.min()}")
            print(f"Max value in image {idx + 1}: {image.max()}")
            print(f"Mean value in image {idx + 1}: {image.mean()}")

            image = image * 256.0
            image = image.numpy().transpose(1, 2, 0)
            image = np.clip(image, 0, 255).astype(np.uint8)

            print(f"\nImage {idx + 1} shape after transpose: {image.shape}")
            print(f"Min value in image {idx + 1}: {image.min()}")
            print(f"Max value in image {idx + 1}: {image.max()}")
            print(f"Mean value in image {idx + 1}: {image.mean()}\n")

            gt_map = gt_maps[idx].squeeze().cpu().numpy()
            pred_map = outputs[idx].squeeze().cpu().numpy()

            # Compute the crowd count from the model predicted density maps: --->
            # img_gt_count = torch.sum(torch.tensor(gt_map)).item()
            # img_pred_count = torch.sum(torch.tensor(pred_map)).item()
            img_gt_count = torch.sum(gt_maps[idx], dim = [0, 1, 2]).item()
            img_pred_count = torch.sum(outputs[idx], dim = [0, 1, 2]).item()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(image)
            axs[0].set_title("Input Image")

            axs[1].imshow(gt_map, cmap='jet')
            axs[1].set_title(f"Ground Truth (Count: {img_gt_count:.1f})")

            axs[2].imshow(pred_map, cmap='jet')
            axs[2].set_title(f"Prediction (Count: {img_pred_count:.1f})")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()

            plt.savefig(f"{VISUAL_ROOT}/ts_sample_{idx + 1}_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_2.png")
            plt.close()

            idx += 1
            del image
            del gt_map
            del pred_map
            del img_gt_count
            del img_pred_count

            print(f"Saved visualization as: {VISUAL_ROOT}/ts_sample_{idx + 1}_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_2.png.")

        break
