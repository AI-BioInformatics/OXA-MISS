import torch
import torch.nn as nn

class LinearProbe2(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=2):
        super(LinearProbe2, self).__init__()
        self.batchnorm_1d = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        patch = x["patch_features"]
        patch = patch.to(torch.float32)
        patch = patch.flatten(start_dim=1)
        patch = self.batchnorm_1d(patch)
        hidden = self.fc1(patch)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
        return {'output': output}
