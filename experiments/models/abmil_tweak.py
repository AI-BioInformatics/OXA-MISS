import torch 
import torch.nn as nn
import torch.nn.functional as F

def select_high_variance_patches_pytorch(embeddings, num_selected_patches):
    """
    Select patches with the highest variance in their embeddings using PyTorch.

    Parameters:
    embeddings (torch.Tensor): Tensor of shape (1, patch_len, hidden_dim).
    num_selected_patches (int): Number of patches to select.

    Returns:
    selected_indices (torch.Tensor): Indices of the selected patches.
    selected_embeddings (torch.Tensor): Embeddings of the selected patches with shape (1, num_selected_patches, hidden_dim).
    """
    # Remove batch dimension
    embeddings = embeddings.squeeze(0)  # Shape: (patch_len, hidden_dim)
    # Compute variance of each embedding vector across its elements
    variances = embeddings.var(dim=1, unbiased=False)  # Shape: (patch_len,)
    # Get indices of embeddings sorted by variance in descending order
    sorted_indices = torch.argsort(variances, descending=True)
    # Select the top 'num_selected_patches' patches with highest variance
    selected_indices = sorted_indices[:num_selected_patches]  # Shape: (num_selected_patches,)
    # Gather selected embeddings
    selected_embeddings = embeddings[selected_indices]  # Shape: (num_selected_patches, hidden_dim)
    # Add batch dimension back
    selected_embeddings = selected_embeddings.unsqueeze(0)  # Shape: (1, num_selected_patches, hidden_dim)
    return selected_indices, selected_embeddings

def select_high_entropy_patches_pytorch(embeddings, num_selected_patches):
    """
    Select patches with the highest entropy in their embeddings using PyTorch.

    Parameters:
    embeddings (torch.Tensor): Tensor of shape (1, patch_len, hidden_dim).
    num_selected_patches (int): Number of patches to select.

    Returns:
    selected_indices (torch.Tensor): Indices of the selected patches.
    selected_embeddings (torch.Tensor): Embeddings of the selected patches with shape (1, num_selected_patches, hidden_dim).
    """
    # Remove batch dimension
    embeddings = embeddings.squeeze(0)  # Shape: (patch_len, hidden_dim)
    # Shift embeddings to make all elements positive
    min_vals, _ = embeddings.min(dim=1, keepdim=True)
    embeddings_shifted = embeddings - min_vals + 1e-8  # Shape: (patch_len, hidden_dim)
    # Normalize embeddings to sum to 1 (convert to probability distributions)
    embeddings_prob = embeddings_shifted / embeddings_shifted.sum(dim=1, keepdim=True)  # Shape: (patch_len, hidden_dim)
    # Compute entropy of each embedding vector
    entropy_values = -torch.sum(embeddings_prob * torch.log(embeddings_prob + 1e-8), dim=1)  # Shape: (patch_len,)
    # Get indices of embeddings sorted by entropy in descending order
    sorted_indices = torch.argsort(entropy_values, descending=True)
    # Select the top 'num_selected_patches' patches with highest entropy
    selected_indices = sorted_indices[:num_selected_patches]  # Shape: (num_selected_patches,)
    # Gather selected embeddings
    selected_embeddings = embeddings[selected_indices]  # Shape: (num_selected_patches, hidden_dim)
    # Add batch dimension back
    selected_embeddings = selected_embeddings.unsqueeze(0)  # Shape: (1, num_selected_patches, hidden_dim)
    return selected_indices, selected_embeddings

class ABMIL_Tweak(nn.Module):
    def __init__(self,input_dim=1024,
                 inner_dim=64, 
                 output_dim=4, 
                 use_layernorm=False, 
                 dropout=0.0):
        super(ABMIL_Tweak,self).__init__()
        
        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

        # Output layer
        self.output_layer = nn.Linear(inner_dim, output_dim)
        
    def forward(self, data):
        # Extract patch features
        x = data['patch_features']  # x is a dictionary with key 'patch_features'
        mask = data['mask']
        x = x[~mask.bool()].unsqueeze(0)
        x = self.inner_proj(x)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        A_out = attn_scores
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, inner_dim)
        weighted_sum = self.dropout(weighted_sum)
        
        # Output layer
        logits = self.output_layer(weighted_sum)  # Shape: (batch_size, output_dim)
        
        output = {'output': logits, 'attention': A_out}
        return output