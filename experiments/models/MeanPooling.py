import torch

import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self, d_model=1024, output_dim=4):
        super(MeanPooling, self).__init__()
        self.fc = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        x = x["patch_features"]
        batch_size, seq_len, d_model = x.size()
        x = x.mean(dim=1)
        output = self.fc(x)
        return {'output': output}