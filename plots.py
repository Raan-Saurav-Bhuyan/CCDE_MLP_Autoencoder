# Import libraries
import os
import matplotlib.pyplot as plt

# Import constants
from const import PLOT_ROOT, TRAIN_TOTAL_LOSS, TRAIN_MSE_LOSS, TRAIN_MAE_LOSS, TEST_TOTAL_LOSS, TEST_MSE_LOSS, TEST_MAE_LOSS

# Helper to generate separate plots for each loss: --->
def save_plot(y_values, x_label, y_label, title, save_path, color = 'black', marker = None):
    plt.figure(figsize = (10, 6))
    plt.plot(y_values, label = title, color = color, marker = marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")

# Train loss plotting: --->
# Function parameters: --->
#   (1) total_losses = Average epoch-wise MSE + MAE losses.
#   (2) mse_losses = Average epoch-wise MSE losses.
#   (3) mae_losses = Average epoch-wise MAE losses.
def train_loss_plots(total_losses, mse_losses, mae_losses):
    os.makedirs(PLOT_ROOT, exist_ok=True)
    # epochs = list(range(1, len(total_losses) + 1))

    # Plot the MSE (Density Map) + MAE (Crowd Count) losses: --->
    save_plot(
        y_values = total_losses,
        x_label = 'Epoch',
        y_label = 'Loss',
        title = 'Train Total Loss (MSE (Density Map) + MAE (Crowd Count))',
        save_path = TRAIN_TOTAL_LOSS,
        color = 'black',
        marker = '^'
    )

    # Plot the MSE (Density Map) losses: --->
    save_plot(
        y_values = mse_losses,
        x_label = 'Epoch',
        y_label = 'MSE Loss',
        title = 'Training MSE (Density Map) Loss',
        save_path = TRAIN_MSE_LOSS,
        color = 'red',
        marker = 'o'
    )

    # Plot the MAE (Crowd Count) losses: --->
    save_plot(
        y_values = mae_losses,
        x_label = 'Epoch',
        y_label = 'MAE Loss',
        title = 'Training MAE (Crowd Count) Loss',
        save_path = TRAIN_MAE_LOSS,
        color = 'blue',
        marker = 's'
    )

# Test loss plotting: --->
# Function parameters: --->
#   (1) batch_losses = Total batch-wise MSE + MAE losses.
#   (2) batch_mse = Total batch-wise MSE losses.
#   (3) batch_mae = Total batch-wise MAE losses.
def test_loss_plot(batch_losses, batch_mse, batch_mae):
    os.makedirs(PLOT_ROOT, exist_ok=True)

    # Plot the MSE (Density Map) + MAE (Crowd Count) losses: --->
    save_plot(
        y_values = batch_losses,
        x_label = 'Test Sample Batch Index',
        y_label = 'Loss',
        title = 'Test Total Loss (MSE (Density Map) + MAE (Crowd Count))',
        save_path = TEST_TOTAL_LOSS,
        color = 'black'
    )

    # Plot the MSE (Density Map) losses: --->
    save_plot(
        y_values = batch_mse,
        x_label = 'Test Sample Batch Index',
        y_label = 'MSE Loss',
        title = 'Test MSE (Density Map) Loss',
        save_path = TEST_MSE_LOSS,
        color = 'red',
        marker = 'o'
    )

    # Plot the MAE (Crowd Count) losses: --->
    save_plot(
        y_values = batch_mae,
        x_label = 'Test Sample Batch Index',
        y_label = 'MAE Loss',
        title = 'Test MAE (Crowd Count) Loss',
        save_path = TEST_MAE_LOSS,
        color = 'blue',
        marker = 's'
    )
