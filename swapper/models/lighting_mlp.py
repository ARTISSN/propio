import torch
import torch.nn as nn

class LightingMLP(nn.Module):
    def __init__(self, input_dim=9, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, lighting_tensor):
        return self.model(lighting_tensor)
