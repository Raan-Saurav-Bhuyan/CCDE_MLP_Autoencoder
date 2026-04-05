# Import libraries: --->
import torch.nn as nn

class CrowdMLP(nn.Module):
    def __init__(self, in_channels=1):
        super(CrowdMLP, self).__init__()

        input_dim = 224 * 224 * in_channels

        # Encoder structure in sequential layers: --->
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        # Decoder structure in sequential layers: --->
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224 * 224),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.encoder(x)
        x = self.decoder(x)

        # reshape to (batch, channel, height, width): --->
        x = x.view(-1, 1, 224, 224)

        return x
