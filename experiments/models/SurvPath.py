import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum

# NUM_PATHWAYS = 1280

def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        
        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        out_pathways =  attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x





# Blocco helper usato da SurvPath
def SNN_Block(dim1, dim2, dropout=0.25):
    """
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )

class SurvPathOriginal(nn.Module):
    """
    Questa è la classe SurvPath originale, mantenuta quasi invariata.
    Verrà usata internamente dalla nostra classe Adapter.
    """
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names=[],
    ):
        super(SurvPathOriginal, self).__init__()
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout

        if omic_names:
            self.omic_names = omic_names
            # (logica per captum rimossa per chiarezza, non necessaria per il forward)

        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.wsi_projection_net = nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim)

        self.init_per_path_model(omic_sizes)

        self.identity = nn.Identity()
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways
        )

        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes)
        )

    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def forward(self, **kwargs):
        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]
        return_attn = kwargs.get("return_attn", False) # .get() per sicurezza
        mask = None

        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)

        wsi_embed = self.wsi_projection_net(wsi)

        # Assicuriamoci che i tensori siano sullo stesso device
        if h_omic_bag.device != wsi_embed.device:
             wsi_embed = wsi_embed.to(h_omic_bag.device)
             
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask, return_attention=False)

        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        paths_postSA_embed = torch.mean(mm_embed[:, :self.num_pathways, :], dim=1)
        wsi_postSA_embed = torch.mean(mm_embed[:, self.num_pathways:, :], dim=1)

        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
        logits = self.to_logits(embedding)

        if return_attn:
            # NOTA: Le attenzioni restituite qui hanno una struttura diversa da quelle del tuo modello.
            # L'adapter non le restituirà per evitare confusione.
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits


# ==========================================================================================
# NUOVA CLASSE ADATTATORE
# ==========================================================================================

class SurvPath(nn.Module):
    """
    Adapter per il modello SurvPath.
    Questa classe agisce come un "drop-in replacement" per Custom_Multimodal_XA,
    accettando lo stesso formato di input e restituendo un output compatibile.

    Gestisce solo le modalità WSI e Genomics e solleva un errore se una delle due manca.
    """
    def __init__(self,
                 input_dim=1024,
                 genomics_group_name=["group1", "group2"],
                 genomics_group_input_dim=[50, 50],
                 output_dim=4,
                 wsi_projection_dim=256, # Parametro specifico di SurvPath
                 dropout=0.1             # Parametro specifico di SurvPath
                 ):
        """
        Inizializza l'adattatore.

        Args:
            input_dim (int): Dimensione degli embedding dei patch WSI (corrisponde a wsi_embedding_dim di SurvPath).
            genomics_group_name (list): Lista dei nomi dei gruppi di geni (usata per accedere ai dati).
            genomics_group_input_dim (list): Lista delle dimensioni di input per ciascun gruppo di geni (corrisponde a omic_sizes di SurvPath).
            output_dim (int): Numero di classi di output (corrisponde a num_classes di SurvPath).
            wsi_projection_dim (int): Dimensione della proiezione per gli embedding WSI e genomici in SurvPath.
            dropout (float): Tasso di dropout per SurvPath.
        """
        super(SurvPath, self).__init__()

        # Conserva i nomi dei gruppi per accedervi nel forward
        self.genomics_group_name = genomics_group_name

        # Istanzia il modello SurvPath originale con i parametri mappati
        self.survpath_model = SurvPathOriginal(
            omic_sizes=genomics_group_input_dim,
            wsi_embedding_dim=input_dim,
            num_classes=output_dim,
            wsi_projection_dim=wsi_projection_dim,
            omic_names=genomics_group_name,
            dropout=dropout
        )

    def forward(self, data):
        """
        Esegue il forward pass.

        Args:
            data (dict): Un dizionario che contiene i dati, con la stessa struttura
                         usata da Custom_Multimodal_XA. Deve contenere:
                         - 'WSI_status': (bool) True se i dati WSI sono presenti.
                         - 'genomics_status': (bool) True se i dati genomici sono presenti.
                         - 'patch_features': (Tensor) Embedding dei patch WSI.
                         - 'mask': (Tensor) Maschera per i patch.
                         - 'genomics': (dict) Dizionario con i dati genomici, dove le chiavi
                                       corrispondono a `genomics_group_name`.
        
        Returns:
            dict: Un dizionario con la chiave 'output' contenente i logits,
                  per compatibilità con il tuo codice.
        """
        # 1. Validazione dell'input: Assicura che entrambe le modalità siano presenti
        if not data.get("WSI_status", False) or not data.get("genomics_status", False):
            raise ValueError("SurvPathAdapter richiede che sia i dati WSI che quelli genomici siano presenti.")

        # 2. Pre-processamento dei dati WSI
        patch_embeddings = data['patch_features']
        mask = data['mask']
        # Applica la maschera e calcola l'embedding aggregato (media) per la WSI
        valid_patches = patch_embeddings[~mask.bool()]
        if valid_patches.shape[0] == 0:
            # Gestisce il caso in cui non ci siano patch validi
            wsi_embedding = torch.zeros(1, self.survpath_model.wsi_embedding_dim, device=patch_embeddings.device)
        else:
            wsi_embedding = torch.mean(valid_patches, dim=0, keepdim=True) # Shape: [1, input_dim]

        # 3. Pre-processamento dei dati Genomici
        genomics_data = data["genomics"]
        # Crea una lista di tensori nel formato richiesto da SurvPath
        try:
            x_omic_list = [genomics_data[name] for name in self.genomics_group_name]
        except KeyError as e:
            raise KeyError(f"Il gruppo di geni '{e.args[0]}' non è stato trovato nel dizionario 'data[\"genomics\"]'. Assicurati che `genomics_group_name` corrisponda alle chiavi.")

        # 4. Costruzione degli argomenti per SurvPath e chiamata al modello
        kwargs = {
            'x_path': wsi_embedding,
            'return_attn': False
        }
        for i, omic_tensor in enumerate(x_omic_list):
            kwargs[f'x_omic{i+1}'] = omic_tensor

        logits = self.survpath_model(**kwargs)

        # 5. Formattazione dell'output per compatibilità
        # SurvPath non ha un output di attenzione strutturato come il tuo modello,
        # quindi restituiamo None per le altre chiavi per evitare errori nel tuo codice.
        output = {
            'output': logits,
            'attention': None,
            'XA_attentions': None
        }

        return output