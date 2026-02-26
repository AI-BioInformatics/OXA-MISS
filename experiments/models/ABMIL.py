import torch 
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self,d_model=1024,output_dim=4,input_dim=1024,
                 genomics_group_name = [ "tumor_suppression", "oncogenesis","protein_kinases", "cellular_differentiation","cytokines_and_growth"],
                    genomics_group_input_dim = [82, 313, 496, 331, 427],
                    genomics_group_dropout =   [0.35],
                    cnv_group_name = [ "tumor_suppression", "oncogenesis","protein_kinases", "cellular_differentiation","cytokines_and_growth"],
                    cnv_group_input_dim = [25, 35, 31],
                    cnv_group_dropout =   [0.2],
                    inner_dim=256, 
                    num_latent_queries=2,
                    wsi_dropout=0,
                    use_layernorm=False, 
                    dropout=0.5,
                    input_modalities = ["WSI", "Genomics"],
                    fusion_type="sum",
                                        use_WSI_level_embs = False,
                    WSI_level_embs_fusion_type = "concat" ,
                    WSI_level_encoder_dropout = 0.2,
                    WSI_level_encoder_sizes = [768, 40, 3],
                    WSI_level_encoder_LayerNorm = False ,
                    ):
        super(ABMIL,self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.attention_V = nn.Linear(input_dim, input_dim)
        self.attention_U = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(input_dim, 1)

        # Output layer
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Extract patch features
        x = x['patch_features']  # x is a dictionary with key 'patch_features'
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: con(batch_size, num_patches, d_model)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, d_model)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, d_model)
        
        # Output layer
        output = self.output_layer(weighted_sum)  # Shape: (batch_size, output_dim)
        output = {'output': output}
        return output