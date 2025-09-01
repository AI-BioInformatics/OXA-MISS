import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
torch.autograd.set_detect_anomaly(True)

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

class OXA_MISS(nn.Module):
    def __init__(self, 
                    input_dim=1024, 
                    genomics_group_name = [ "tumor_suppression", "oncogenesis","protein_kinases", "cellular_differentiation","cytokines_and_growth"],
                    genomics_group_input_dim = [82, 313, 496, 331, 427],
                    genomics_group_dropout =   [0.35],
                    cnv_group_name = [ "tumor_suppression", "oncogenesis","protein_kinases", "cellular_differentiation","cytokines_and_growth"],
                    cnv_group_input_dim = [25, 35, 31],
                    cnv_group_dropout =   [0.2],
                    inner_dim=256, 
                    output_dim=4, 
                    num_latent_queries=2,
                    wsi_dropout=0,
                    use_layernorm=False, 
                    dropout=0.5,
                    input_modalities = ["WSI", "Genomics"],
                    fusion_type="sum",
                    
                    use_WSI_level_embs= False,
                    WSI_level_embs_fusion_type= "concat", # sum | concat
                    WSI_level_encoder_dropout= 0.2,
                    WSI_level_encoder_sizes= [768, 60, 10],
                    WSI_level_encoder_LayerNorm= False,
                    ):
        super(OXA_MISS,self).__init__()
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
            

        if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
            cnv = data["cnv"]
            cnv_groups = []
            for key in self.cnv_group_name:
                cnv_group_i = cnv[key]
                cnv_group_i = self.cnv_encoder[key](cnv_group_i)
                cnv_groups.append(cnv_group_i)
            cnv_embedding = torch.stack(cnv_groups, dim=1)

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
                patch_embeddings_updated = patch_embeddings + x  
            else:
                patch_embeddings_updated = patch_embeddings
            if "CNV" in self.input_modalities and data["cnv_status"].item() is True:
                patch_embeddings_updated = patch_embeddings + y
            else:
                patch_embeddings_updated = patch_embeddings
            XA_attentions["att_patches_to_genomics"] = att_patches_to_genomics.detach() if att_patches_to_genomics is not None else None
            XA_attentions["att_patches_to_cnv"] =  att_patches_to_cnv.detach() if att_patches_to_cnv is not None else None
            
            keys = self.W_k(patch_embeddings_updated)
            scores = torch.matmul(latent_queries, keys.transpose(1, 2))
            scores /= torch.sqrt(torch.tensor(keys.size(-1)).float())
            scores = gate.transpose(-1,-2) * scores 
            scores = self.wsi_dropout(scores)
            A_out = scores
            scores = F.softmax(scores, dim=-1)
            latent = torch.matmul(scores,patch_embeddings_updated)
            latent = latent.flatten(start_dim=1)

            #Extract high level features
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