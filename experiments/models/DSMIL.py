import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, d_model=128, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, d_model)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, input_dim, output_dim,
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
        super(DSMIL, self).__init__()
        self.i_classifier = FCLayer(input_dim, output_dim)
        self.b_classifier = BClassifier(input_size=input_dim, output_class=output_dim)
        
    def forward(self, x):
        x = x["patch_features"]
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats.squeeze(dim=0), classes.squeeze(dim=0))
        
        # return classes, prediction_bag, A, B
        return {'output': prediction_bag}
        