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
from loss import CompositeCrowdLoss
from plots import test_loss_plot

# Import constants: --->
from const import DEVICE, DATASET_ROOT, BATCH_SIZE,  MODEL, CSV_ROOT, CSV_NAME, GRAY

# Set the color channels of the input dimension: --->
if GRAY:
    CHANNEL = 1
elif not GRAY:
    CHANNEL = 3
else:
    print(f"The value of GRAY = {GRAY} is invalid!")
    quit()

def test_model():
    # Load the test subset: --->
    test_dataset = CrowdDataset(DATASET_ROOT, split = 'test')
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # Load THE trained model: --->
    model = CrowdMLP(in_channels = CHANNEL).to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location = DEVICE))
    model.eval()

    # Instantiate loss class object: --->
    criterion = CompositeCrowdLoss()

    # Check whether the directory exists or not: --->
    os.makedirs(CSV_ROOT, exist_ok = True)

    # Initialization of metric history: --->
    all_pred_counts = []
    all_gt_counts = []
    batch_losses = []
    batch_mse = []
    batch_mae = []

    with torch.no_grad():
        for idx, (inputs, gt_maps) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            gt_maps = gt_maps.to(DEVICE)

            # Save the predictions of the model over the batch of images and compute the losses: --->
            outputs = model(inputs)
            loss, mse, mae = criterion(outputs, gt_maps)

            # Compute the crowd count from the model predicted density maps (per image): --->
            pred_counts = torch.sum(outputs, dim = [1, 2, 3])
            gt_counts = torch.sum(gt_maps, dim = [1, 2, 3])

            # Store per-image counts and batch-wise losses: --->
            all_pred_counts.extend(pred_counts.cpu().numpy())
            all_gt_counts.extend(gt_counts.cpu().numpy())

            batch_losses.append(loss.item())
            batch_mse.append(mse.item())
            batch_mae.append(mae.item())

            print(f"[Batch {idx + 1}/{len(test_loader)}] Loss: {loss.item():.4f} | MSE Loss: {mse.item():.4f} | MAE Loss: {mae.item():.4f}")

    # Save per-image comparison table as CSV file: --->
    comparison_df = pd.DataFrame({
        "Image_Index": list(range(1, len(all_pred_counts)+1)),
        "Predicted_Count": all_pred_counts,
        "Ground_Truth_Count": all_gt_counts,
    })

    comparison_df.to_csv(CSV_NAME, index=False)

    print("\n==== Test Summary ====")
    test_loss_plot(batch_losses, batch_mse, batch_mae)

    print(f"Saved comparison table as {CSV_NAME}.")
