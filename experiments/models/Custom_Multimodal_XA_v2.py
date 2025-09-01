import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
torch.autograd.set_detect_anomaly(True)

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

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, context): # original forward
        B, N, C = x.shape

        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

    # def forward(self, x, context):
    #     # x: (B, N, C)
    #     B, N, C = x.shape

    #     # Compute queries, keys, and values.
    #     q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    #     k = self.key(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    #     v = self.value(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    #     # Create a scale tensor in a reproducible way.
    #     scale_tensor = torch.tensor(self.scale, dtype=q.dtype, device=q.device)

    #     # Compute the attention scores with matrix multiplication.
    #     scores = torch.matmul(q, k.transpose(-2, -1)) * scale_tensor

    #     # Subtract the maximum score per row for numerical stability
    #     # This ensures the same summation order when computing the exponentials.
    #     max_scores = scores.max(dim=-1, keepdim=True)[0]
    #     scores = scores - max_scores

    #     # Compute the deterministic softmax manually.
    #     exp_scores = torch.exp(scores)
    #     attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    #     # Apply dropout (ensure that your dropout is seeded deterministically)
    #     attn = self.attn_drop(attn)

    #     # Compute the weighted sum over the values.
    #     x = torch.matmul(attn, v)
    #     x = x.transpose(1, 2).reshape(B, N, C)

    #     # Apply the projection and dropout.
    #     x = self.proj(x)
    #     x = self.proj_drop(x)

    #     return x, attn
    
class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Custom_Multimodal_XA_v2(nn.Module):
    def __init__(self, 
                    input_dim=1024, 
                    genomics_group_name = ["high_refractory", "high_sensitive", "hypoxia_pathway"],
                    genomics_group_input_dim = [25, 35, 31],
                    genomics_group_dropout =   [0.2, 0.2, 0.2],
                    cnv_group_name = ["high_refractory", "high_sensitive", "hypoxia_pathway"],
                    cnv_group_input_dim = [25, 35, 31],
                    cnv_group_dropout =   [0.2, 0.2, 0.2],
                    inner_dim=64, 
                    output_dim=4, 
                    num_latent_queries=4,
                    wsi_dropout=0.2,
                    use_layernorm=False, 
                    dropout=0.0,
                    input_modalities = ["WSI", "Genomics", "CNV"],
                    fusion_type="sum", # "concatenate" or "sum"
                    
                    use_WSI_level_embs= True, # False True 
                    WSI_level_embs_fusion_type= "concat", # sum | concat
                    WSI_level_encoder_dropout= 0.3,
                    WSI_level_encoder_sizes= [768, 60, 10],
                    WSI_level_encoder_LayerNorm= True,
                    ):
        super(Custom_Multimodal_XA_v2,self).__init__()
        self.input_modalities = input_modalities
        self.inner_proj = nn.Linear(input_dim, inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.num_latent_queries = num_latent_queries
        self.use_layernorm = use_layernorm
        self.fusion_type = fusion_type
        self.genomics_group_name = genomics_group_name
        if ('Genomics' in input_modalities and len(genomics_group_input_dim) != len(genomics_group_name)) or ('CNV' in input_modalities and len(cnv_group_input_dim) != len(cnv_group_name)):
            raise ValueError("Mismatch between genomics_group_name or cnv_group_name  and genomics_group_input_dim lengths.")
        self.genomics_group_input_dim = genomics_group_input_dim
        if 'Genomics' in input_modalities and len(genomics_group_dropout) != len(genomics_group_name):
            if len(genomics_group_dropout) == 1:
                genomics_group_dropout = genomics_group_dropout * len(genomics_group_name)
            else:
                raise ValueError("Mismatch between genomics_group_name and genomics_group_dropout lengths.")
        self.genomics_group_dropout = genomics_group_dropout
        self.cnv_group_name = cnv_group_name
        self.cnv_group_input_dim = cnv_group_input_dim
        if 'CNV' in input_modalities and len(cnv_group_dropout) != len(cnv_group_name):
            if len(cnv_group_dropout) == 1:
                cnv_group_dropout = cnv_group_dropout * len(cnv_group_name)
            else:
                raise ValueError("Mismatch between cnv_group_name and cnv_group_dropout lengths.")
        self.cnv_group_dropout = cnv_group_dropout
        self.wsi_dropout = nn.Dropout(wsi_dropout)
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

        self.patches_XA = CrossAttentionBlock(inner_dim)
        self.patches_FF = FeedForwardLayer(inner_dim, inner_dim, inner_dim)
        self.genomics_XA = CrossAttentionBlock(inner_dim)
        self.genomics_FF = FeedForwardLayer(inner_dim, inner_dim, inner_dim)
        self.cnv_XA = CrossAttentionBlock(inner_dim)
        self.cnv_FF = FeedForwardLayer(inner_dim, inner_dim, inner_dim)


        # self.layernorm_wsi_embeddings = nn.LayerNorm(inner_dim)
        # self.layernorm_genomics_embedding= nn.LayerNorm(inner_dim)
        # self.layernorm_cnv_embedding = nn.LayerNorm(inner_dim)

        if "Genomics" in self.input_modalities:
            self.genomic_encoder = {}
            for name, input_dim, rate in zip(genomics_group_name, genomics_group_input_dim, genomics_group_dropout):
                self.genomic_encoder[name] = nn.Sequential(
                                                nn.Dropout(rate),
                                                nn.Linear(input_dim, inner_dim),
                                                nn.ReLU(),
                                                nn.Linear(inner_dim, inner_dim),
                                            ) 
            self.genomic_encoder = nn.ModuleDict(self.genomic_encoder)

        if "CNV" in self.input_modalities:
            self.cnv_encoder = {}
            for name, input_dim, rate in zip(cnv_group_name, cnv_group_input_dim, cnv_group_dropout):
                self.cnv_encoder[name] = nn.Sequential(
                                                nn.Dropout(rate),
                                                nn.Linear(input_dim, inner_dim),
                                                nn.ReLU(),
                                                nn.Linear(inner_dim, inner_dim),
                                            )
            self.cnv_encoder = nn.ModuleDict(self.cnv_encoder)

        # Output layer
        if fusion_type == "concatenate":
            final_layer_input_dim = 0
            if "WSI" in input_modalities:
                final_layer_input_dim += inner_dim
            if "Genomics" in input_modalities:
                final_layer_input_dim += inner_dim
            if "CNV" in input_modalities:
                final_layer_input_dim += inner_dim
        elif fusion_type == "sum":
            final_layer_input_dim = inner_dim
        else:
            raise ValueError("Invalid fusion type. Choose between 'concatenate' or 'sum'.")
            
        self.output_layer = nn.Linear(final_layer_input_dim, output_dim)

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
        if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
            patch_embeddings = data['patch_features'] 
            mask = data['mask']
            patch_embeddings = patch_embeddings[~mask.bool()].unsqueeze(0)
            patch_embeddings = self.inner_proj(patch_embeddings)
            
            if self.use_layernorm:
                patch_embeddings = self.layernorm(patch_embeddings)  
                latent_queries = self.layernorm_latent(self.latent_queries)      
            else:
                latent_queries = self.latent_queries 
            
            # Apply attention mechanism
            gate = self.sigmoid(self.gate(patch_embeddings))

            
        if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
            genomics = data["genomics"]
            genomics_groups = []
            for key in self.genomics_group_name:
                genomics_group_i = genomics[key]
                genomics_group_i = self.genomic_encoder[key](genomics_group_i)
                genomics_groups.append(genomics_group_i)           
            genomics_embedding = torch.stack(genomics_groups, dim=1)
            

            # genomics_embedding = self.layernorm_genomics_embedding(genomics_embedding)

        if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
            cnv = data["cnv"]
            cnv_groups = []
            for key in self.cnv_group_name:
                cnv_group_i = cnv[key]
                cnv_group_i = self.cnv_encoder[key](cnv_group_i)
                cnv_groups.append(cnv_group_i)
            cnv_embedding = torch.stack(cnv_groups, dim=1)

            # cnv_embedding = self.layernorm_cnv_embedding(cnv_embedding)

        XA_attentions = {}
        if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
            if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
                x, att_patches_to_genomics = self.patches_XA(patch_embeddings, genomics_embedding)
            else:
                att_patches_to_genomics = None
            if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
                y, att_patches_to_cnv = self.patches_XA(patch_embeddings, cnv_embedding)
            else:
                att_patches_to_cnv = None
            if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
                # patch_embeddings = patch_embeddings + x   
                patch_embeddings_updated = patch_embeddings + x  
            else:
                patch_embeddings_updated = patch_embeddings
            if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
                # patch_embeddings = patch_embeddings + y 
                patch_embeddings_updated = patch_embeddings + y
            else:
                patch_embeddings_updated = patch_embeddings
            XA_attentions["att_patches_to_genomics"] = att_patches_to_genomics.detach() if att_patches_to_genomics is not None else None
            XA_attentions["att_patches_to_cnv"] =  att_patches_to_cnv.detach() if att_patches_to_cnv is not None else None
            
            # x = self.patches_FF(patch_embeddings)
            # patch_embeddings = patch_embeddings + x
            
            # keys = self.W_k(patch_embeddings)
            keys = self.W_k(patch_embeddings_updated)
            scores = torch.matmul(latent_queries, keys.transpose(1, 2))
            scores /= torch.sqrt(torch.tensor(keys.size(-1)).float())
            scores = gate.transpose(-1,-2) * scores # scores + log(gate+eps)
            scores = self.wsi_dropout(scores)
            A_out = scores
            scores = F.softmax(scores, dim=-1)
            # latent = torch.matmul(scores, patch_embeddings)
            latent = torch.matmul(scores,patch_embeddings_updated)
            latent = latent.flatten(start_dim=1)

            #Extract high level features
            # latent = self.tanh(latent)
            wsi_embedding = self.fc(latent)

        if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
            if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
                x, att_genomics_to_patches = self.genomics_XA(genomics_embedding, patch_embeddings)
            else:
                att_genomics_to_patches = None
            if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
                y, att_genomics_to_cnv = self.genomics_XA(genomics_embedding, cnv_embedding)
            else:
                att_genomics_to_cnv = None
            if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
                genomics_embedding = genomics_embedding + x
            if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
                genomics_embedding = genomics_embedding + y
            XA_attentions["att_genomics_to_patches"] = att_genomics_to_patches.detach() if att_genomics_to_patches is not None else None
            XA_attentions["att_genomics_to_cnv"] = att_genomics_to_cnv.detach() if att_genomics_to_cnv is not None else None

            # x = self.genomics_FF(genomics_embedding)
            # genomics_embedding = genomics_embedding + x
            genomics_embedding = genomics_embedding.sum(dim=1, keepdim=False)


        if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
            if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
                x, att_cnv_to_patches = self.cnv_XA(cnv_embedding, patch_embeddings)
            else:
                att_cnv_to_patches = None
            if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
                y, att_cnv_to_genomics = self.cnv_XA(cnv_embedding, genomics_embedding)
            else:
                att_cnv_to_genomics = None
            if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
                cnv_embedding = cnv_embedding + x
            if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
                cnv_embedding = cnv_embedding + y
            XA_attentions["att_cnv_to_patches"] = att_cnv_to_patches.detach() if att_cnv_to_patches is not None else None
            XA_attentions["att_cnv_to_genomics"] = att_cnv_to_genomics.detach() if att_cnv_to_genomics is not None else None

            # x = self.cnv_FF(cnv_embedding)
            # cnv_embedding = cnv_embedding + x
            cnv_embedding = cnv_embedding.sum(dim=1, keepdim=False)

        modalities = []
        if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
            modalities.append(wsi_embedding)
        if "Genomics" in self.input_modalities and data["genomics_status"].item() is True:
            modalities.append(genomics_embedding)
        if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
            modalities.append(cnv_embedding)

        if self.fusion_type == "sum":
            x = sum(modalities)
        elif self.fusion_type == "concatenate":
            x = torch.cat(modalities, dim=1)
        else:
            raise ValueError("Invalid fusion type. Choose between 'concatenate' or 'sum'.")

        # Output layer
        x = self.dropout(x)
        logits = self.output_layer(x)  # Shape: (batch_size, output_dim)
        
        if "WSI" in self.input_modalities and data["WSI_status"].item() is True:
            output = {'output': logits, 'attention': A_out, 'XA_attentions': XA_attentions}
        else:
            output = {'output': logits, 'XA_attentions': XA_attentions}
        return output