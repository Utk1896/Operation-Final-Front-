import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphPredictor(nn.Module):
    def __init__(self, input_channels=1, max_nodes=10):
        super(GraphPredictor, self).__init__()
        self.max_nodes = max_nodes
        self.output_size = max_nodes * max_nodes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
            nn.Sigmoid()  # Output in [0,1] for each adjacency element
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out.view(-1, self.max_nodes, self.max_nodes)
