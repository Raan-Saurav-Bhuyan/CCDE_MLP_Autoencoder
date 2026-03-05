# Import libraries: --->
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchinfo import summary

# Import local modules: --->
from model import CrowdMLP
from dataset import CrowdDataset
from loss import CompositeCrowdLoss
from plots import train_loss_plots

# Import constants: --->
from const import DEVICE, DATASET_ROOT, BATCH_SIZE, LR, EPOCHS, MODEL_ROOT, MODEL, GRAY

if GRAY:
    CHANNEL = 1
elif not GRAY:
    CHANNEL = 3
else:
    print(f"The value of GRAY = {GRAY} is invalid!")
    quit()

def train_model():
    # Load the train subset: --->
    train_dataset = CrowdDataset(DATASET_ROOT, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Make the directory for the model saving: --->
    os.makedirs(MODEL_ROOT, exist_ok=True)

    # Instantiate model, loss and optimizer: --->
    model = CrowdMLP().to(DEVICE)
    criterion = CompositeCrowdLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    summary(model, (1, CHANNEL, 224, 224))

    # Initialization of epoch-wise  metric history: --->
    epoch_losses = []
    mse_losses = []
    mae_losses = []
    min_prev = None

    for epoch in range(EPOCHS):
        model.train()

        # Initialize the epoch-wise count of losses: --->
        running_loss = 0.0
        running_mse = 0.0
        running_mae = 0.0

        # Loop through the train subset batches of images: --->
        for inputs, gt_maps in train_loader:
            # Map the objects into the device (either CPU or GPU): --->
            inputs = inputs.to(DEVICE)
            gt_maps = gt_maps.to(DEVICE)

            # Forward pass: --->
            outputs = model(inputs)
            loss, mse, mae = criterion(outputs, gt_maps)

            # Back-propagate: --->
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the loss per batch in each epoch: --->
            running_loss += loss.item() * inputs.size(0)
            running_mse += mse.item() * inputs.size(0)
            running_mae += mae.item() * inputs.size(0)

        # Compute epoch metrics: --->
        avg_loss = running_loss / len(train_loader.dataset)
        avg_mse = running_mse / len(train_loader.dataset)
        avg_mae = running_mae / len(train_loader.dataset)

        if epoch > 0:
            min_prev = min(epoch_losses)
        epoch_losses.append(avg_loss)
        mse_losses.append(avg_mse)
        mae_losses.append(avg_mae)

        print(f"[Epoch {epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | MSE Loss: {avg_mse:.4f} | MAE Loss: {avg_mae:.4f}")

        # Save the best model out of all epochs: --->
        min_loss = min(epoch_losses)
        if avg_loss <= min_loss:
            torch.save(model.state_dict(), MODEL)
            print(f"Loss reduced from {min_prev} to {avg_loss}. Model saved to {MODEL}.")

    print("\n==== Train Summary ====")
    # Plot the losses as metrics for the training of the model: --->
    train_loss_plots(epoch_losses, mse_losses, mae_losses)
