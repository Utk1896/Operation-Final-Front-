import torch
import torch.nn as nn
from .advanced_unet import AdvancedUNet

class GraphPredictor(nn.Module):
    def __init__(self, max_nodes=10):
        super().__init__()
        self.encoder = AdvancedUNet(in_channels=1, out_features=max_nodes * max_nodes)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, max_nodes * max_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
