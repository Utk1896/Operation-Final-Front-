import torch
import torch.nn as nn

class GraphExtractor(nn.Module):
    def __init__(self):
        super(GraphExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halve spatial dims
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halve again
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # halve again
        )

        # Dynamically get the flatten size for your input image size (e.g. 128x128 or 64x64)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)  # replace 128,128 with your actual input size
            dummy_output = self.conv_layers(dummy_input)
            flatten_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100)  # output size: max_nodes*max_nodes (10*10=100)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
