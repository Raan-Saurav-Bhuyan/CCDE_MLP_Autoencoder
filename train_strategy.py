# Import libraries: --->
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
import numpy as np

# Import local modules: --->
from model import CrowdMLP
from dataset import CrowdDataset
from loss import CompositeCrowdLoss
from plots import train_loss_plots

# Import constants: --->
from const import DEVICE, DATASET_ROOT, BATCH_SIZE, LR, EPOCHS, MODEL_ROOT, MODEL, GRAY

# Set the color channels of the input dimension: --->
if GRAY:
    CHANNEL = 1
elif not GRAY:
    CHANNEL = 3
else:
    print(f"The value of GRAY = {GRAY} is invalid!")
    quit()

def train_model():
    # Load the full dataset and prepare batches: --->
    full_dataset = CrowdDataset(DATASET_ROOT, split='train')
    total_samples = len(full_dataset)
    indices = list(range(total_samples))

    # Shuffle the indices of the training dataset: --->
    np.random.shuffle(indices)

    # Create list of initial random batch indices: --->
    K = total_samples // BATCH_SIZE
    batches = [indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(K)]

    # Make the directory for the model saving: --->
    os.makedirs(MODEL_ROOT, exist_ok=True)

    # Create model, loss, optimizer, and view the model summary: --->
    model = CrowdMLP(in_channels=CHANNEL).to(DEVICE)
    criterion = CompositeCrowdLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    summary(model, (1, CHANNEL, 224, 224))

    # Initialize the epoch and loss record lists: --->
    epoch_losses = []
    mse_losses = []
    mae_losses = []
    min_prev = None
    avg_loss = 0
    avg_mse = 0
    avg_mae = 0

    # Iterative training over remaining epochs and batches:
    for epoch in range(EPOCHS):
        batch_losses = []
        print(f"\n=== Epoch {epoch + 1} / {EPOCHS} ===")

        # Train initial batch (batch index 0):
        initial_batch = Subset(full_dataset, batches[0])
        initial_loader = DataLoader(initial_batch, batch_size = BATCH_SIZE, shuffle = True)

        for i in range(5):
            model.train()

            for inputs, gt_maps in initial_loader:
                inputs = inputs.to(DEVICE)
                gt_maps = gt_maps.to(DEVICE)

                outputs = model(inputs)
                loss, mse, mae = criterion(outputs, gt_maps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for iter in range(K):
            # Test on all batches except the last trained one: --->
            for i, batch_indices in enumerate(batches):
                # skip the already trained batch: --->
                if batch_indices == batches[0]:
                    continue

                test_subset = Subset(full_dataset, batch_indices)
                test_loader = DataLoader(test_subset, batch_size = BATCH_SIZE, shuffle = False)

                model.eval()
                running_loss = 0.0

                with torch.no_grad():
                    for inputs, gt_maps in test_loader:
                        inputs = inputs.to(DEVICE)
                        gt_maps = gt_maps.to(DEVICE)

                        outputs = model(inputs)
                        loss, _, _ = criterion(outputs, gt_maps)
                        running_loss += loss.item() * inputs.size(0)

                avg_loss = running_loss / len(test_loader.dataset)
                batch_losses.append((i, avg_loss))

            # Sort batch losses descending and derive the worst performing batch: --->
            batch_losses.sort(key=lambda x: x[1], reverse = True)
            worst_batch_index = batch_losses[0][0]
            print(f"Training on worst batch: {worst_batch_index} | Loss: {batch_losses[0][1]:.4f}")

            # Train on worst batch: --->
            model.train()
            train_subset = Subset(full_dataset, batches[worst_batch_index])
            train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle = True)

            running_loss = 0.0
            running_mse = 0.0
            running_mae = 0.0

            for inputs, gt_maps in train_loader:
                inputs = inputs.to(DEVICE)
                gt_maps = gt_maps.to(DEVICE)

                outputs = model(inputs)
                loss, mse, mae = criterion(outputs, gt_maps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_mse += mse.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)

            avg_loss = running_loss / len(train_loader.dataset)
            avg_mse = running_mse / len(train_loader.dataset)
            avg_mae = running_mae / len(train_loader.dataset)

        if epoch > 0:
            min_prev = min(epoch_losses)
        epoch_losses.append(avg_loss)
        mse_losses.append(avg_mse)
        mae_losses.append(avg_mae)

        print(f"\nLoss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f}")

        # Save best model: --->
        # min_loss = min(epoch_losses)
        # if avg_loss <= min_loss:
        torch.save(model.state_dict(), MODEL)
        print(f"Loss changed from {min_prev} to {avg_loss}. Model saved to {MODEL}.")

        # Shuffle indices of the samples for next epoch: --->
        np.random.shuffle(indices)
        batches = [indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(K)]

    print("\n==== Training Complete ====")
    train_loss_plots(epoch_losses, mse_losses, mae_losses)
