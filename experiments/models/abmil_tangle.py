import torch 
import torch.nn as nn
import torch.nn.functional as F

class ABMIL_Tangle(nn.Module):
    def __init__(self,d_model=1024,output_dim=4):
        super(ABMIL_Tangle,self).__init__()
        self.fc = nn.Linear(d_model,output_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.attention_V = nn.Linear(d_model, d_model)
        self.attention_U = nn.Linear(d_model, d_model)
        self.attention_weights = nn.Linear(d_model, 1)

        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Extract patch features
        x = x['patch_features']  # x is a dictionary with key 'patch_features'
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, d_model)
        U = torch.tanh(self.attention_U(x))  # Shape: (batch_size, num_patches, d_model)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, d_model)
        
        # Output layer
        output = self.output_layer(weighted_sum)  # Shape: (batch_size, output_dim)
        
        return output