import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayerNormalization, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x