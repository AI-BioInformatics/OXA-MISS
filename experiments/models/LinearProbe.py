import torch

import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        self.batchnorm_1d = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        patch = x["patch_features"]
        patch = patch.to(torch.float32)
        patch = patch.flatten(start_dim=1)
        patch = self.batchnorm_1d(patch)
        output = self.fc(patch)
        return {'output': output}