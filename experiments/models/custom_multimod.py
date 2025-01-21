import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class Custom_Multimod(nn.Module):
    def __init__(self, 
                    input_dim=1024, 
                    inner_dim=64, 
                    output_dim=4, 
                    num_latent_queries=4,
                    use_layernorm=False, 
                    dropout=0.0
                    ):
        super(Custom_Multimod,self).__init__()

        self.inner_proj = nn.Linear(input_dim, inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.num_latent_queries = num_latent_queries
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
            self.layernorm_latent = nn.LayerNorm(inner_dim)
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, inner_dim))
        self.W_k = nn.Linear(inner_dim, inner_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gate = nn.Linear(inner_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(num_latent_queries*inner_dim, inner_dim)

        # Output layer           
        self.output_layer = nn.Linear(inner_dim, output_dim)

        # Initialize latent queries
        # init.kaiming_normal_(self.latent_queries , mode='fan_in', nonlinearity='relu')
        # Initialize latent queries with identity matrix
        # with torch.no_grad():
        #     # Use the built-in identity initialization for the weight
        #     torch.nn.init.eye_(self.W_k.weight)
        #     # Set the bias to zero
        #     torch.nn.init.zeros_(self.W_k.bias)
        
    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)  

    def forward(self, data):
        # Extract patch features
        x = data['patch_features']  # x is a dictionary with key 'patch_features'
        mask = data['mask']
        x = x[~mask.bool()].unsqueeze(0)
        x = self.inner_proj(x)
        
        if self.use_layernorm:
            x = self.layernorm(x)  
            latent_queries = self.layernorm_latent(self.latent_queries)      
        else:
            latent_queries = self.latent_queries 
        
        # Apply attention mechanism
        gate = self.sigmoid(self.gate(x))
        keys = self.W_k(x)
        scores = torch.matmul(latent_queries, keys.transpose(1, 2))
        scores /= torch.sqrt(torch.tensor(keys.size(-1)).float())
        scores = gate.transpose(-1,-2) * scores
        A_out = scores
        scores = F.softmax(scores, dim=-1)
        latent = torch.matmul(scores, x)
        latent = latent.flatten(start_dim=1)

        #Extract high level features
        # latent = self.tanh(latent)
        latent = self.fc(latent)
        latent = self.relu(latent)
        latent = self.dropout(latent)
        
        # Output layer
        logits = self.output_layer(latent)  # Shape: (batch_size, output_dim)
        
        output = {'output': logits, 'attention': A_out}
        return output