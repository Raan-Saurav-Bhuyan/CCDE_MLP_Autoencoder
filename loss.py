# Import libraries: --->
import torch
import torch.nn as nn

class CompositeCrowdLoss(nn.Module):
    def __init__(self):
        super(CompositeCrowdLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred_density, gt_density):
        # Density Map Loss (MSE): --->
        mse_loss = self.mse(pred_density, gt_density)

        # Crowd Count Loss (MAE between summed counts): --->
        pred_count = torch.sum(pred_density, dim = [1, 2, 3])  # sum over [B, 1, H, W]
        gt_count = torch.sum(gt_density, dim = [1, 2, 3])
        mae_loss = self.mae(pred_count, gt_count)

        total_loss = mse_loss + mae_loss

        return total_loss, mse_loss, mae_loss
