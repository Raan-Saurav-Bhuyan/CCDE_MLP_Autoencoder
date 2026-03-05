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
from const import DEVICE, DATASET_ROOT, BATCH_SIZE,  MODEL, VISUAL_ROOT, CSV_ROOT, CSV_NAME

def test_model():
    # Load the test subset: --->
    test_dataset = CrowdDataset(DATASET_ROOT, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load THE trained model: --->
    model = CrowdMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()

    # Instantiate loss class object: --->
    criterion = CompositeCrowdLoss()

    # Check whether the directory exists or not: --->
    os.makedirs(CSV_ROOT, exist_ok=True)

    # Initialization of batch-wise metric history: --->
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

            # Compute the crowd count from the model predicted density maps: --->
            pred_count = torch.sum(outputs).item()
            gt_count = torch.sum(gt_maps).item()

            # Store batch metrics: --->
            all_pred_counts.append(pred_count)
            all_gt_counts.append(gt_count)

            batch_losses.append(loss.item())
            batch_mse.append(mse.item())
            batch_mae.append(mae.item())

            print(f"[Batch {idx + 1}/{int(len(test_loader.dataset) / BATCH_SIZE) + 1}] Loss: {loss.item():.4f} | MSE Loss: {mse.item():.4f} | MAE Loss: {mae.item():.4f}")

    # Save Comparison Table as CSV file: --->
    comparison_df = pd.DataFrame({
        "Image_Index": list(range(1, len(all_pred_counts)+1)),
        "Predicted_Count": all_pred_counts,
        "Ground_Truth_Count": all_gt_counts,
        "MSE_Loss": batch_mse,
        "MAE_Loss": batch_mae,
        "Total_Loss": batch_losses
    })

    comparison_df.to_csv(CSV_NAME, index=False)

    print("\n==== Test Summary ====")
    test_loss_plot(batch_losses, batch_mse, batch_mae)

    print(f"Saved comparison table as {CSV_NAME}.")
