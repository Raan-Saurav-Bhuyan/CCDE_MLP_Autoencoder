# Import libraries: --->
import torch.nn as nn

class CrowdMLP(nn.Module):
    def __init__(self):
        super(CrowdMLP, self).__init__()

        # Encoder structure in sequential layers: --->
        self.encoder = nn.Sequential(
            # nn.Linear(224 * 224 * 3, 1024),            #! Experiment 1 (RGB image as input of size 224 x 224 x 3)
            nn.Linear(224 * 224 * 1, 1024),            #! Experiment 2 (Monochrome image as input of size 224 x 224 x 1)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.encoder(x)
        x = self.decoder(x)

        # reshape to (batch, channel, height, width): --->
        x = x.view(-1, 1, 224, 224)

        return x
