"""
OXA_MISS_final: sostituto di OXA_MISS (nome definitivo da scegliere). Stessa interfaccia di OXA_MISS.py
(chiave 'output', xa_fusion, return_xa_attentions, ct/mri_encoder, fusione 'sum'/'concatenate'), con:

  1. CLINICA a token al posto dell'MLP su 14 feature:
       ClinicalFeatureTokenizer (un token per variabile) -> [cross-attention] -> media dei token presenti
       -> LayerNorm
     Input in `data`: clinical_status, clinical_num (B,1) GREZZO, clinical_num_mask (B,1), clinical_cat (B,K)
     al posto di clinical_features (dataloader/clinical_tokens.py, attivato da data_loader.clinical).
     La maschera dei token e' per paziente: ogni paziente usa solo le proprie variabili osservate.
  2. Standardizzazione dell'eta' DENTRO il modello: media/std dei pazienti di training del fold, salvate come
     buffer (set_clinical_normalization, chiamato da main.py per ogni fold): il checkpoint e' autosufficiente.
  3. Cross-attention fra WSI, Genomica e Clinica (clinical_xa=True, default; False = solo WSI <-> Genomica
     come OXA_MISS, per ablation), con UN blocco per ogni coppia (query <- contesto): ogni blocco ha le sue
     proiezioni Q/K/V per il proprio contesto.
  4. Scala indipendente dalle modalita' presenti: con xa_fusion ogni modalita' viene aggiornata con la MEDIA
     degli aggiornamenti dei contesti disponibili, seguita da LayerNorm (x <- LN(x + media(update))) anche
     quando nessun contesto e' disponibile; la sua norma non dipende quindi da quante modalita' ha il paziente.
  5. SENZA CNV: "CNV" in input_modalities solleva errore.
  6. Missingness clinica: i token mancanti sono esclusi da attenzioni e media; in training se ne spengono
     anche a caso alcuni presenti (clinical_token_dropout), lasciandone almeno uno per paziente. Senza token
     osservati la clinica e' assente (slot a zero in 'concatenate', come per ogni modalita' mancante).
  7. Pooling WSI con gated attention (Ilse et al. 2018) per le latent queries: chiavi tanh(V h) * sigmoid(U h)
     calcolate sulle patch DOPO la cross-attention (gate e chiavi sulle stesse patch), dropout sui pesi di
     attenzione dopo la softmax (OXA_MISS: sigmoide moltiplicata sui logit, che li spinge verso l'uniforme;
     dropout sui logit, che li azzera invece di escluderli; gate sulle patch prima della cross-attention).
  8. Flag di ablation (default = modello proposto; docs/design_rationale.md): wsi_gated_keys,
     genomics_single_token, xa_shared_blocks, xa_update (mean_ln | sum), clinical_missing (mask | impute |
     indicator), clinical_xa, xa_fusion. Con i valori di OXA_MISS ne riproducono le scelte.

Batch: la parte clinica e' per paziente, ma il ramo WSI seleziona le patch con la maschera di un solo
paziente (come OXA_MISS): il modello richiede ancora batch di un paziente (data_loader.batch_size: 1).
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, context, context_mask=None):
        """x (B,N,C) queries, context (B,M,C); context_mask (B,M) bool, True = valid token of the patient.
        A patient without valid context tokens gets a zero update."""
        B, N, C = x.shape

        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        empty = None
        if context_mask is not None:
            empty = ~context_mask.any(dim=1)                          # (B,) patients without context
            valid = context_mask | empty.unsqueeze(1)                 # avoid an all -inf row (NaN)
            attn = attn.masked_fill(~valid[:, None, None, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if empty is not None and empty.any():
            x = x * (~empty).to(x.dtype)[:, None, None]
        return x, attn


class GatedAttentionPooling(nn.Module):
    """K learned queries pool a set of tokens with gated keys tanh(V h) * sigmoid(U h) (Ilse et al., 2018),
    dropout on the attention weights after the softmax: (B, N, dim) -> (B, K, dim). Trained end-to-end with
    the model (CT / MRI regions)."""
    def __init__(self, dim, num_queries, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * dim ** -0.5)
        self.V = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        keys = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))              # (B, N, dim)
        scores = torch.matmul(self.queries, keys.transpose(1, 2)) / keys.size(-1) ** 0.5   # (B, K, N)
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, tokens), scores


class RadiologyHub(nn.Module):
    """Radiology in the cross-attention: the core vector (fused WSI / Genomics / Clinical) is the query, the
    pooled CT / MRI tokens of the patient the context. Each modality's tokens get a content gate (scalar per
    patient, from their mean) and a modality-type embedding; the available ones are concatenated.
    core <- LN1(core + XA(core, tokens)); core <- LN2(core + FF(core)). Without radiology the update is zero but
    the same LN / FF path runs: the scale of the output does not depend on the presence of CT / MRI."""
    def __init__(self, dim, modalities, dropout=0.1):
        super().__init__()
        self.modalities = list(modalities)
        self.type_embedding = nn.Embedding(len(self.modalities), dim)
        self.gates = nn.ModuleDict({m: nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
                                    for m in self.modalities})
        self.cross_attention = CrossAttentionBlock(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, core, tokens):
        """core (B, dim); tokens {modality: (B, K, dim)} of the available radiology -> (B, dim), attention or None."""
        pieces = []
        for i, m in enumerate(self.modalities):
            if m in tokens:
                gate = torch.sigmoid(self.gates[m](tokens[m].mean(dim=1)))              # (B, 1)
                pieces.append(tokens[m] * gate.unsqueeze(1) + self.type_embedding.weight[i])
        attention = None
        if pieces:
            update, attention = self.cross_attention(core.unsqueeze(1), torch.cat(pieces, dim=1))
            core = core + update.squeeze(1)
        core = self.norm1(core)
        return self.norm2(core + self.ff(core)), attention


class GradientReversal(torch.autograd.Function):
    """Identity forward, gradient multiplied by -1 backward (Ganin et al., JMLR 2016)."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


# sex, ajcc_stage, ajcc_t, ajcc_n, ajcc_m (vocabolario in dataloader/clinical_tokens.py; main.py passa le
# cardinalita' del dataset in clinical_cat_cardinalities)
DEFAULT_CAT_CARDINALITIES = [2, 4, 4, 4, 2]


class ClinicalFeatureTokenizer(nn.Module):
    """Un token per variabile (stile FT-Transformer), con maschera di presenza per paziente.
    - numeriche: valore * scala + bias appresi per variabile; arrivano GREZZE: z-score con num_mean / num_std
      (buffer: media/std del training del fold, salvate nel checkpoint);
    - categoriche: embedding con padding_idx=0 (mancante = vettore nullo, non appreso);
    - valori mancanti (missing, ablation): "mask" li toglie (token_mask False); "impute" li sostituisce con
      media / categoria piu' frequente del training del fold (buffer); "indicator" con un token "mancante"
      appreso per variabile. Con impute / indicator tutti i token sono presenti;
    - ritorna (tokens (B,T,dim), token_mask (B,T) bool: True = token usato per quel paziente)."""
    def __init__(self, n_numerical, cat_cardinalities, dim, missing="mask"):
        super().__init__()
        if missing not in ("mask", "impute", "indicator"):
            raise ValueError(f"clinical_missing {missing}: mask | impute | indicator")
        self.missing = missing
        self.n_numerical, self.n_categorical = n_numerical, len(cat_cardinalities)
        self.register_buffer("num_mean", torch.zeros(n_numerical))
        self.register_buffer("num_std", torch.ones(n_numerical))
        self.register_buffer("cat_mode", torch.ones(len(cat_cardinalities), dtype=torch.long))
        # buffer (in the state_dict): a checkpoint knows that its statistics were set
        self.register_buffer("normalization_set", torch.tensor(False))
        if missing == "indicator":
            self.missing_tokens = nn.Parameter(torch.randn(n_numerical + len(cat_cardinalities), dim) * 0.02)
        self.num_scale = nn.Parameter(torch.randn(n_numerical, dim) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_numerical, dim))
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim, padding_idx=0) for card in cat_cardinalities])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, num, cat, num_mask):
        """num (B,n_num) float grezzo | cat (B,n_cat) long, 0=mancante | num_mask (B,n_num) bool"""
        num_mask = num_mask.bool()
        observed = torch.cat([num_mask, cat > 0], dim=1)            # (B, T) bool, per paziente
        missing = getattr(self, "missing", "mask")                  # checkpoints older than the flag
        if missing == "impute":                                # media / moda del training del fold
            num = torch.where(num_mask, num, self.num_mean.expand_as(num))
            cat = torch.where(cat > 0, cat, self.cat_mode.expand_as(cat))
            num_mask = torch.ones_like(num_mask)
        num = (num - self.num_mean) / self.num_std
        num_tokens = num.unsqueeze(-1) * self.num_scale + self.num_bias
        num_tokens = num_tokens * num_mask.unsqueeze(-1)            # mancante -> vettore nullo
        cat_tokens = torch.stack([e(cat[:, i]) for i, e in enumerate(self.cat_embeddings)], dim=1)
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)         # (B, n_num+n_cat, dim)
        if missing == "mask":
            return tokens, observed
        if missing == "indicator":
            tokens = torch.where(observed.unsqueeze(-1), tokens, self.missing_tokens.unsqueeze(0).expand_as(tokens))
        return tokens, torch.ones_like(observed)


class OXA_MISS_final(nn.Module):
    def __init__(self,
                    input_dim=1024,
                    genomics_group_name = [ "tumor_suppression", "oncogenesis","protein_kinases", "cellular_differentiation","cytokines_and_growth"],
                    genomics_group_input_dim = [82, 313, 496, 331, 427],
                    genomics_group_dropout =   [0.35],
                    inner_dim=256,
                    output_dim=4,
                    num_latent_queries=2,
                    wsi_dropout=0,
                    use_layernorm=False,
                    dropout=0.5,
                    input_modalities = ["WSI", "Genomics"],
                    fusion_type="concatenate",  # sum | concatenate
                    ct_emb_dim=2048,                # size of the CT / MRI region tokens (set by main.py)
                    mri_emb_dim=2048,
                    num_ct_latent_queries=4,        # summarizing tokens of the CT attention pooling
                    num_mri_latent_queries=2,       # MRI: fewer (the scarcest modality)
                    radiology_pooling="attention",  # attention: gated attention pooling | mean: one token = mean of the
                                                    # projected regions = projection of the mean-pooled exam vector (ablation)
                    radiology_dropout=0.3,          # training: each available CT / MRI dropped with this probability
                    multilevel_loss=True,           # survival loss on every radiology configuration + consistency
                    multilevel_aux_weight=0.3,      # weight of the loss of the smaller configurations
                    multilevel_consistency_weight=0.1,  # KL(parent configuration || richer configuration)
                    xa_fusion=True,
                    clinical_n_numerical=1,                     # eta'
                    clinical_cat_cardinalities=None,            # None -> DEFAULT_CAT_CARDINALITIES (sex, stage, T, N, M)
                    clinical_token_dropout=0.15,
                    clinical_xa=True,                           # clinica nella cross-attention (False: ablation, come OXA_MISS)
                    # --- ablation flags (default = modello proposto) ---
                    wsi_gated_keys=True,            # False: chiavi lineari W h (come OXA_MISS)
                    genomics_single_token=False,    # True: un solo MLP su tutti i geni (un token) invece di uno per gruppo
                    xa_shared_blocks=False,         # True: un blocco per modalita' query, condiviso fra i contesti (OXA_MISS)
                    xa_graph="full",                 # full: ogni coppia WSI/Genomica/Clinica | wsi_star: solo coppie con la WSI
                                                    # (genomica <-> clinica non si parlano direttamente)
                    clinical_fusion="core",         # core: la clinica entra nel vettore core (come ora) | late_logit: testa clinica
                                                    # separata, sommata ai logit (la clinica non passa dal core ne' dall'hub)
                    xa_update="mean_ln",            # mean_ln: LN(x + media degli update) | sum: x + somma (OXA_MISS)
                    clinical_missing="mask",        # mask | impute | indicator
                    # --- module A: probabilistic fusion (fusion_type="poe") ---
                    poe_kl_weight=1e-3,             # KL(posterior || N(0, I)) in the loss
                    poe_subset_dropout=0.3,         # training: each available expert dropped with this probability (>= 1 kept)
                    poe_predictor_weight=0.1,       # loss of the cross-modal predictors (value of acquisition)
                    poe_samples=20,                 # Monte-Carlo samples of the posterior for the risk uncertainty
                    # --- module B: shortcut removal (adversary with gradient reversal) ---
                    shortcut_adversary_weight=0.0,  # 0 = off
                    shortcut_adversary_target="study",    # study (cohort) | site (TCGA tissue source site) | pattern (missingness) | both (study + pattern)
                    n_studies=1,                    # set by main.py from the dataset
                    n_cancer_types=1,               # set by main.py from the dataset
                    n_sites=1,                      # set by main.py from the dataset
                    ):
        super(OXA_MISS_final,self).__init__()
        if "CNV" in input_modalities:
            raise ValueError("OXA_MISS_final non supporta CNV: rimuoverlo da input_modalities.")
        self.input_modalities = input_modalities
        # xa_fusion (default True): the cross-attention outputs update the patch / genomics / clinical
        # embeddings used for the prediction. False = previous model, where the cross-attention outputs were
        # discarded: it is then computed only when its attention maps are requested (return_xa_attentions,
        # set by ModelManager to save them).
        self.xa_fusion = xa_fusion
        self.return_xa_attentions = False
        self.inner_proj = nn.Linear(input_dim, inner_dim)
        self.use_layernorm = use_layernorm
        self.fusion_type = fusion_type
        self.genomics_group_name = genomics_group_name
        if 'Genomics' in input_modalities and len(genomics_group_input_dim) != len(genomics_group_name):
            raise ValueError("Mismatch between genomics_group_name and genomics_group_input_dim lengths.")
        if 'Genomics' in input_modalities and len(genomics_group_dropout) != len(genomics_group_name):
            if len(genomics_group_dropout) == 1:
                genomics_group_dropout = genomics_group_dropout * len(genomics_group_name)
            else:
                raise ValueError("Mismatch between genomics_group_name and genomics_group_dropout lengths.")
        self.wsi_dropout = nn.Dropout(wsi_dropout)
        self.dropout = nn.Dropout(dropout)

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
            self.layernorm_latent = nn.LayerNorm(inner_dim)
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, inner_dim))
        # gated attention keys: tanh(V h) * sigmoid(U h); wsi_gated_keys=False: linear keys V h
        self.wsi_gated_keys = wsi_gated_keys
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        if wsi_gated_keys:
            self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.fc = nn.Linear(num_latent_queries*inner_dim, inner_dim)

        # cross-attention: one block per (query <- context) pair of the token modalities, and one LayerNorm per
        # query modality applied after the residual update (scale independent of the available contexts)
        self.clinical_xa = clinical_xa and 'Clinical' in input_modalities
        xa_modalities = [m for m, name in (("patches", "WSI"), ("genomics", "Genomics"), ("clinical", "Clinical"))
                         if name in input_modalities and (name != "Clinical" or self.clinical_xa)]
        self.xa_pairs = [(q, c) for q in xa_modalities for c in xa_modalities if q != c]
        if xa_graph not in ("full", "wsi_star"):
            raise ValueError(f"xa_graph {xa_graph}: full | wsi_star")
        self.xa_graph = xa_graph
        if xa_graph == "wsi_star":     # WSI al centro: genomica e clinica scambiano informazione solo con le patch
            self.xa_pairs = [(q, c) for q, c in self.xa_pairs if "patches" in (q, c)]
        if xa_update not in ("mean_ln", "sum"):
            raise ValueError(f"xa_update {xa_update}: mean_ln | sum")
        self.xa_update = xa_update
        self.xa_shared_blocks = xa_shared_blocks
        block_names = sorted({q for q, _ in self.xa_pairs}) if xa_shared_blocks else [f"{q}_to_{c}" for q, c in self.xa_pairs]
        self.xa_blocks = nn.ModuleDict({name: CrossAttentionBlock(inner_dim) for name in block_names})
        if xa_update == "mean_ln":
            self.xa_norms = nn.ModuleDict({q: nn.LayerNorm(inner_dim) for q in sorted({q for q, _ in self.xa_pairs})})

        # radiology: region tokens -> projection -> gated attention pooling (K summarizing tokens), trained
        # end-to-end; they enter the model through the radiology hub (or as PoE experts)
        if clinical_fusion not in ("core", "late_logit"):
            raise ValueError(f"clinical_fusion {clinical_fusion}: core | late_logit")
        if clinical_fusion == "late_logit" and fusion_type == "poe":
            raise ValueError("clinical_fusion=late_logit non e' supportato con fusion_type poe")
        self.clinical_fusion = clinical_fusion
        self.core_modalities = [m for m in ("WSI", "Genomics", "Clinical") if m in input_modalities
                                and not (m == "Clinical" and clinical_fusion == "late_logit")]
        self.radiology_modalities = [m for m in ("CT", "MRI") if m in input_modalities]
        self.radiology_dropout = radiology_dropout
        rad_dims = {"CT": ct_emb_dim, "MRI": mri_emb_dim}
        rad_queries = {"CT": num_ct_latent_queries, "MRI": num_mri_latent_queries}
        self.radiology_proj = nn.ModuleDict({m: nn.Linear(rad_dims[m], inner_dim) for m in self.radiology_modalities})
        if radiology_pooling not in ("attention", "mean"):
            raise ValueError(f"radiology_pooling {radiology_pooling}: attention | mean")
        self.radiology_pooling = radiology_pooling
        if radiology_pooling == "attention":
            self.radiology_pool = nn.ModuleDict({m: GatedAttentionPooling(inner_dim, rad_queries[m])
                                                 for m in self.radiology_modalities})
        if 'Clinical' in self.input_modalities:
            if clinical_cat_cardinalities is None:
                clinical_cat_cardinalities = DEFAULT_CAT_CARDINALITIES
            self.clinical_token_dropout = clinical_token_dropout
            self.clinical_tokenizer = ClinicalFeatureTokenizer(
                clinical_n_numerical, clinical_cat_cardinalities, inner_dim, missing=clinical_missing)
            self.clinical_norm = nn.LayerNorm(inner_dim)
        self.genomics_single_token = genomics_single_token
        if "Genomics" in self.input_modalities and genomics_single_token:
            # one MLP over all the genes of the groups (PORPOISE-style): a single genomics token
            self.genomic_encoder = nn.ModuleDict({"all": nn.Sequential(
                                                nn.Dropout(genomics_group_dropout[0]),
                                                nn.Linear(sum(genomics_group_input_dim), inner_dim),
                                                nn.ReLU(),
                                                nn.Linear(inner_dim, inner_dim),
                                            )})
        elif "Genomics" in self.input_modalities:
            self.genomic_encoder = {}
            for name, group_dim, rate in zip(genomics_group_name, genomics_group_input_dim, genomics_group_dropout):
                self.genomic_encoder[name] = nn.Sequential(
                                                nn.Dropout(rate),
                                                nn.Linear(group_dim, inner_dim),
                                                nn.ReLU(),
                                                nn.Linear(inner_dim, inner_dim),
                                            )
            self.genomic_encoder = nn.ModuleDict(self.genomic_encoder)

        # core vector (WSI / Genomics / Clinical) -> radiology hub -> head on inner_dim
        if fusion_type not in ("concatenate", "sum", "poe"):
            raise ValueError("Invalid fusion type. Choose between 'concatenate', 'sum' or 'poe'.")
        if fusion_type == "concatenate" and self.core_modalities:
            # [e_WSI | e_Genomics | e_Clinical] (zeros for a missing one) -> inner_dim
            self.core_compress = nn.Linear(inner_dim * len(self.core_modalities), inner_dim)
        # radiology hub with concatenate / sum; with poe CT / MRI are experts of the product (value of acquisition)
        self.uses_hub = bool(self.radiology_modalities) and fusion_type != "poe"
        self.radiology_hub = RadiologyHub(inner_dim, self.radiology_modalities) if self.uses_hub else None
        self.multilevel_loss = multilevel_loss and self.uses_hub and fusion_type != "poe"
        self.multilevel_aux_weight, self.multilevel_consistency_weight = multilevel_aux_weight, multilevel_consistency_weight
        if fusion_type == "poe":
            # module A: one Gaussian expert per modality (mean, log-variance of a diagonal Gaussian), a learned
            # prior expert, closed-form product of experts over the available ones (MVAE, Wu & Goodman 2018),
            # and per modality a predictor of its expert from the posterior of the others (value of acquisition)
            self.poe_kl_weight, self.poe_subset_dropout = poe_kl_weight, poe_subset_dropout
            self.poe_predictor_weight, self.poe_samples = poe_predictor_weight, poe_samples
            # experts: the core modalities and CT / MRI (mean of their pooled tokens)
            self.poe_modalities = self.core_modalities + self.radiology_modalities
            self.poe_experts = nn.ModuleDict({m: nn.Linear(inner_dim, 2 * inner_dim) for m in self.poe_modalities})
            self.poe_prior_mu = nn.Parameter(torch.zeros(inner_dim))
            self.poe_prior_logvar = nn.Parameter(torch.zeros(inner_dim))
            n_mod = len(self.poe_modalities)
            self.poe_predictors = nn.ModuleDict({m: nn.Sequential(
                nn.Linear(2 * inner_dim + n_mod, inner_dim), nn.ReLU(), nn.Linear(inner_dim, 3 * inner_dim))
                for m in self.poe_modalities})

        self.output_layer = nn.Linear(inner_dim, output_dim)
        if clinical_fusion == "late_logit" and 'Clinical' in input_modalities:
            self.clinical_head = nn.Linear(inner_dim, output_dim)   # contributo additivo (nei logit) della clinica
        self._init_adversary(inner_dim, shortcut_adversary_weight, shortcut_adversary_target, n_studies, n_cancer_types,
                             clinical_n_numerical, clinical_cat_cardinalities or DEFAULT_CAT_CARDINALITIES, n_sites)

    def _init_adversary(self, inner_dim, weight, target, n_studies, n_cancer_types, clinical_n_numerical,
                        clinical_cat_cardinalities, n_sites=1):
        """Module B: an adversary predicts the missingness pattern (which modalities, which clinical variables
        are present) and/or the study from the fused representation through a gradient reversal, so that
        the representation does not encode them. It also gets the cancer type: the invariance is conditional
        on it, real differences between tumours are not removed."""
        self.shortcut_adversary_weight = weight
        self.shortcut_adversary_target = target
        if weight <= 0:
            return
        if target not in ("pattern", "study", "both", "site"):
            raise ValueError(f"shortcut_adversary_target {target}: study | site | pattern | both")
        if self.fusion_type == "concatenate":
            raise ValueError("shortcut adversary with fusion_type concatenate: a missing modality is a slot of exact "
                             "zeros, the pattern cannot be removed; use fusion_type sum or poe")
        self.n_cancer_types, self.n_studies, self.n_sites = n_cancer_types, n_studies, n_sites
        n_clinical = (clinical_n_numerical + len(clinical_cat_cardinalities)) if 'Clinical' in self.input_modalities else 0
        self.n_pattern = len(self.input_modalities) + n_clinical
        n_out = (self.n_pattern if target in ("pattern", "both") else 0) + (n_studies if target in ("study", "both") else 0) \
            + (n_sites if target == "site" else 0)
        self.shortcut_adversary = nn.Sequential(nn.Linear(inner_dim + n_cancer_types, inner_dim), nn.ReLU(),
                                                nn.Linear(inner_dim, n_out))

    def _adversary_loss(self, fused, data, on):
        """Adversary loss on the representation of the head (gradient reversed): BCE on the pattern bits
        (modalities available + clinical variables observed, from the data: not the token mask, which token
        dropout / imputation change), cross-entropy on the study."""
        cancer = F.one_hot(data["cancer_type_index"].long().view(-1), self.n_cancer_types).to(fused.dtype)
        out = self.shortcut_adversary(torch.cat([GradientReversal.apply(fused), cancer], dim=-1))
        loss = 0.0
        if self.shortcut_adversary_target in ("pattern", "both"):
            bits = [torch.full((fused.shape[0], 1), float(on[m]), device=fused.device) for m in self.input_modalities]
            if 'Clinical' in self.input_modalities:
                n_clin = self.n_pattern - len(self.input_modalities)
                if on["Clinical"]:
                    observed = torch.cat([data["clinical_num_mask"].bool(), data["clinical_cat"] > 0], dim=1)
                    bits.append(observed.to(fused.dtype))
                else:
                    bits.append(torch.zeros(fused.shape[0], n_clin, device=fused.device))
            loss = loss + F.binary_cross_entropy_with_logits(out[:, :self.n_pattern], torch.cat(bits, dim=1))
        if self.shortcut_adversary_target in ("study", "both"):
            loss = loss + F.cross_entropy(out[:, -self.n_studies:], data["study_index"].long().view(-1))
        if self.shortcut_adversary_target == "site":   # within a cancer type: the tissue source site (TCGA barcode)
            loss = loss + F.cross_entropy(out, data["site_index"].long().view(-1))
        return self.shortcut_adversary_weight * loss

    def set_clinical_normalization(self, mean, std, cat_mode=None):
        """Media / std delle variabili numeriche (eta') e categoria piu' frequente di ogni variabile categorica
        (solo per clinical_missing=impute) sui pazienti di training del fold."""
        tokenizer = self.clinical_tokenizer
        tokenizer.num_mean.copy_(torch.as_tensor(mean, dtype=tokenizer.num_mean.dtype))
        tokenizer.num_std.copy_(torch.as_tensor(std, dtype=tokenizer.num_std.dtype))
        if cat_mode is not None:
            tokenizer.cat_mode.copy_(torch.as_tensor(cat_mode, dtype=tokenizer.cat_mode.dtype))
        elif tokenizer.missing == "impute":
            raise ValueError("clinical_missing=impute: set_clinical_normalization needs cat_mode")
        tokenizer.normalization_set.fill_(True)

    # ------------------------------------------------------------------ module A: product of experts
    POE_LOGVAR_RANGE = (-6.0, 6.0)   # precision clamp: no single expert can dominate numerically

    def _expert(self, modality, embedding):
        mu, logvar = self.poe_experts[modality](embedding).chunk(2, dim=-1)
        return mu, logvar.clamp(*self.POE_LOGVAR_RANGE)

    def _product(self, experts):
        """Posterior N(mu, var) of the prior expert and the given experts {modality: (mu, logvar)}:
        precision-weighted mean, precisions summed."""
        prior_logvar = self.poe_prior_logvar.clamp(*self.POE_LOGVAR_RANGE)
        precision = torch.exp(-prior_logvar).expand(1, -1)
        weighted = self.poe_prior_mu.expand(1, -1) * precision
        for mu, logvar in experts.values():
            p = torch.exp(-logvar)
            precision = precision + p
            weighted = weighted + mu * p
        return weighted / precision, -torch.log(precision)

    def _availability(self, modalities):
        return torch.tensor([[float(m in modalities) for m in self.poe_modalities]],
                            device=self.poe_prior_mu.device)

    def _predict_expert(self, modality, mu, logvar, others):
        """Predicted distribution of the expert of `modality` given the posterior of the others: Gaussian
        (mean, log-variance) of its mean, and its log-variance."""
        out = self.poe_predictors[modality](torch.cat([mu, logvar, self._availability(others)], dim=-1))
        pred_mu, pred_spread, pred_logvar = out.chunk(3, dim=-1)
        return pred_mu, pred_spread.clamp(*self.POE_LOGVAR_RANGE), pred_logvar.clamp(*self.POE_LOGVAR_RANGE)

    def _risk(self, logits):
        """Same risk as ModelManager.calculate_risk: -sum of the survival curve."""
        return -torch.cumprod(1 - torch.sigmoid(logits), dim=-1).sum(dim=-1)

    def _risk_samples(self, mu, logvar, n):
        z = mu.unsqueeze(0) + torch.exp(0.5 * logvar).unsqueeze(0) * torch.randn(n, *mu.shape, device=mu.device)
        return self._risk(self.output_layer(z))                  # (n, B)

    def _poe_fusion(self, embeddings):
        """embeddings {modality: (B, D)} of the available experts -> (z for the head, aux loss, posterior mean,
        posterior log-variance, risk std)."""
        experts = {m: self._expert(m, e) for m, e in embeddings.items()}
        used = experts
        if self.training and self.poe_subset_dropout > 0 and len(experts) > 1:
            names = list(experts)
            keep = [m for m in names if torch.rand(()) >= self.poe_subset_dropout]
            if not keep:
                keep = [names[int(torch.randint(len(names), ()))]]
            used = {m: experts[m] for m in keep}
        mu, logvar = self._product(used)
        aux_loss, risk_std = None, None
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar).sum(dim=-1).mean()
            # cross-modal predictors: expert of m from the posterior of the other available experts
            # (inputs and targets detached: the predictors do not change the representation)
            pred_loss = 0.0
            for m, (t_mu, t_logvar) in experts.items():
                others = {o: experts[o] for o in experts if o != m}
                o_mu, o_logvar = self._product({o: (a.detach(), b.detach()) for o, (a, b) in others.items()})
                p_mu, p_spread, p_logvar = self._predict_expert(m, o_mu, o_logvar, others)
                nll = 0.5 * (p_spread + (t_mu.detach() - p_mu) ** 2 * torch.exp(-p_spread))
                pred_loss = pred_loss + nll.mean() + ((t_logvar.detach() - p_logvar) ** 2).mean()
            aux_loss = self.poe_kl_weight * kl + self.poe_predictor_weight * pred_loss / max(len(experts), 1)
        else:
            z = mu
            risk_std = self._risk_samples(mu, logvar, self.poe_samples).std(dim=0)
        return z, aux_loss, mu, logvar, risk_std

    @torch.no_grad()
    def value_of_acquisition(self, data, n_outer=16, n_inner=64):
        """For every modality of the model that this patient lacks: expected reduction of the variance of
        the predicted risk if it were acquired, E_{expert ~ predictor}[Var_now - Var_after]. Also returns
        the current risk variance. Uses the cross-modal predictors; call in eval mode."""
        out = self.forward(data)
        embeddings = {m: e for m, e in out["embeddings"].items() if m in self.poe_modalities}
        experts = {m: self._expert(m, e) for m, e in embeddings.items()}
        mu, logvar = self._product(experts)
        var_now = self._risk_samples(mu, logvar, n_inner).var(dim=0)
        values = {}
        for m in self.poe_modalities:
            if m in experts:
                continue
            p_mu, p_spread, p_logvar = self._predict_expert(m, mu, logvar, experts)
            var_after = []
            for _ in range(n_outer):
                sampled = p_mu + torch.exp(0.5 * p_spread) * torch.randn_like(p_mu)
                a_mu, a_logvar = self._product({**experts, m: (sampled, p_logvar)})
                var_after.append(self._risk_samples(a_mu, a_logvar, n_inner).var(dim=0))
            values[m] = (var_now - torch.stack(var_after).mean(dim=0))
        return {"risk_var": var_now, "value": values}

    def _drop_tokens(self, token_mask):
        """Solo in training: spegne a caso token clinici presenti (p=clinical_token_dropout), cosi' il
        modello non si appoggia a una variabile specifica ne' al pattern di quelle presenti.
        Per paziente: a un paziente che ha token osservati ne resta sempre almeno uno."""
        if not self.training or self.clinical_token_dropout <= 0:
            return token_mask
        keep = torch.rand(token_mask.shape, device=token_mask.device) >= self.clinical_token_dropout
        dropped = token_mask & keep
        emptied = ~dropped.any(dim=1) & token_mask.any(dim=1)       # pazienti rimasti senza token
        return torch.where(emptied.unsqueeze(1), token_mask, dropped)

    def forward(self, data):
        if data["WSI_status"].numel() != 1:
            raise ValueError("OXA_MISS_final: batch di un paziente (data_loader.batch_size: 1)")
        # modality availability, read once (each .item() on a GPU tensor is a device sync)
        wsi_on = "WSI" in self.input_modalities and bool(data["WSI_status"].item())
        genomics_on = "Genomics" in self.input_modalities and bool(data["genomics_status"].item())
        ct_on = "CT" in self.input_modalities and bool(data["ct_status"].item())
        mri_on = "MRI" in self.input_modalities and bool(data["mri_status"].item())
        clinical_on = "Clinical" in self.input_modalities and bool(data["clinical_status"].item())
        # Extract patch features
        if wsi_on:
            patch_embeddings = data['patch_features']
            mask = data['mask']
            patch_embeddings = patch_embeddings[~mask.bool()].unsqueeze(0)
            patch_embeddings = self.inner_proj(patch_embeddings)

            if self.use_layernorm:
                patch_embeddings = self.layernorm(patch_embeddings)
                latent_queries = self.layernorm_latent(self.latent_queries)
            else:
                latent_queries = self.latent_queries
        # radiology: region tokens (B, N, dim) -> projection -> gated attention pooling (K tokens), trained
        # end-to-end; in training each available exam is dropped with radiology_dropout, so that its presence
        # is not a reliable signal
        radiology_tokens, radiology_raw = {}, {}
        for modality, on, key in (("CT", ct_on, "ct_features"), ("MRI", mri_on, "mri_features")):
            if not on:
                continue
            raw = data[key].reshape(data[key].shape[0], -1, data[key].shape[-1])
            radiology_raw[modality] = raw
            if self.training and torch.rand(()) < self.radiology_dropout:
                continue
            projected = self.radiology_proj[modality](raw)
            if getattr(self, "radiology_pooling", "attention") == "mean":    # checkpoints older than the flag
                radiology_tokens[modality] = projected.mean(dim=1, keepdim=True)     # (B, 1, dim)
            else:
                radiology_tokens[modality], _ = self.radiology_pool[modality](projected)
        ct_on, mri_on = "CT" in radiology_tokens, "MRI" in radiology_tokens
        if clinical_on:
            if not bool(self.clinical_tokenizer.normalization_set):
                raise RuntimeError("OXA_MISS_final: set_clinical_normalization() non chiamato (media/std del training)")
            # tutti i token (B,T,dim) con la maschera dei valori osservati di ciascun paziente (B,T)
            clinical_tokens, clinical_token_mask = self.clinical_tokenizer(
                data['clinical_num'], data['clinical_cat'], data['clinical_num_mask'])
            clinical_token_mask = self._drop_tokens(clinical_token_mask)
            clinical_on = bool(clinical_token_mask.any())   # nessun valore osservato: clinica assente
        if genomics_on:
            genomics = data["genomics"]
            if getattr(self, "genomics_single_token", False):
                all_genes = torch.cat([genomics[key] for key in self.genomics_group_name], dim=-1)
                genomics_embedding = self.genomic_encoder["all"](all_genes).unsqueeze(1)   # (B, 1, D)
            else:
                genomics_groups = []
                for key in self.genomics_group_name:
                    genomics_group_i = genomics[key]
                    genomics_group_i = self.genomic_encoder[key](genomics_group_i)
                    genomics_groups.append(genomics_group_i)
                genomics_embedding = torch.stack(genomics_groups, dim=1)

        # cross-attention on the pre-fusion tokens of each modality: (tokens, mask of the valid tokens)
        tokens = {}
        if wsi_on:
            tokens["patches"] = (patch_embeddings, None)
        if genomics_on:
            tokens["genomics"] = (genomics_embedding, None)
        if clinical_on and self.clinical_xa:
            tokens["clinical"] = (clinical_tokens, clinical_token_mask)
        # getattr: models pickled before these options existed keep their original behaviour
        xa_fusion = getattr(self, "xa_fusion", False)
        compute_xa = xa_fusion or getattr(self, "return_xa_attentions", False)
        XA_attentions = {}
        updated = {name: x for name, (x, _) in tokens.items()}
        # getattr: checkpoints pickled before the ablation flags existed keep the defaults
        xa_update = getattr(self, "xa_update", "mean_ln")
        if compute_xa:
            for query in tokens:
                updates = []
                for q, context in self.xa_pairs:
                    if q != query or context not in tokens:
                        continue
                    block = self.xa_blocks[q if getattr(self, "xa_shared_blocks", False) else f"{q}_to_{context}"]
                    update, att = block(tokens[query][0], *tokens[context])
                    updates.append(update)
                    XA_attentions[f"att_{query}_to_{context}"] = att.detach()
                if xa_fusion and xa_update == "mean_ln" and query in self.xa_norms:
                    # mean of the updates of the available contexts + LayerNorm, also without contexts:
                    # the scale of the modality does not depend on how many modalities the patient has
                    residual = torch.stack(updates, dim=0).mean(dim=0) if updates else 0
                    updated[query] = self.xa_norms[query](tokens[query][0] + residual)
                elif xa_fusion and xa_update == "sum" and updates:
                    updated[query] = tokens[query][0] + torch.stack(updates, dim=0).sum(dim=0)   # OXA_MISS
            if "clinical" in tokens:
                # variables of the clinical tokens of the att_*clinical* maps (order: numerical, categorical)
                XA_attentions["clinical_token_mask"] = clinical_token_mask.detach()

        if wsi_on:
            pooled_patches = updated["patches"]  # embeddings pooled by the latent queries
            # gate and keys from the same (cross-attention updated) patches
            if getattr(self, "wsi_gated_keys", True):
                keys = torch.tanh(self.attention_V(pooled_patches)) * torch.sigmoid(self.attention_U(pooled_patches))
            else:
                keys = self.attention_V(pooled_patches)
            scores = torch.matmul(latent_queries, keys.transpose(1, 2)) / keys.size(-1) ** 0.5   # (1, Q, N)
            A_out = scores                                   # pre-softmax (AEM entropy loss)
            weights = self.wsi_dropout(F.softmax(scores, dim=-1))
            latent = torch.matmul(weights, pooled_patches)
            latent = latent.flatten(start_dim=1)

            #Extract high level features
            wsi_embedding = self.fc(latent)

        if genomics_on:
            genomics_embedding = updated["genomics"].sum(dim=1, keepdim=False)

        if clinical_on:
            clinical_updated = updated.get("clinical", clinical_tokens)
            # MEDIA sui soli token osservati di ciascun paziente: la scala non dipende da quanti sono
            weights = clinical_token_mask.unsqueeze(-1).to(clinical_updated.dtype)
            clinical_mean = (clinical_updated * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            clinical_embedding = self.clinical_norm(clinical_mean)

        # zero embedding, created directly on the inputs' device (no CPU -> GPU copy)
        missing_embedding = lambda: torch.zeros((data['patch_features'].shape[0], self.inner_proj.out_features),
                                                device=data['patch_features'].device)
        core = {}
        if wsi_on:
            core["WSI"] = wsi_embedding
        if genomics_on:
            core["Genomics"] = genomics_embedding
        if clinical_on and getattr(self, "clinical_fusion", "core") == "core":
            core["Clinical"] = clinical_embedding
        late_clinical = 0       # clinical_fusion=late_logit: stesso termine additivo in ogni configurazione radiologica
        if clinical_on and getattr(self, "clinical_fusion", "core") == "late_logit":
            late_clinical = self.clinical_head(self.dropout(clinical_embedding))

        aux_loss = risk_std = None
        multilevel = None
        if self.fusion_type == "poe":
            # CT / MRI as experts: the mean of their pooled tokens
            experts_in = {**core, **{m: t.mean(dim=1) for m, t in radiology_tokens.items()}}
            x, aux_loss, posterior_mu, posterior_logvar, risk_std = self._poe_fusion(experts_in)
            core_vector = posterior_mu
        else:
            # core vector: "concatenate" -> [e_WSI | e_Genomics | e_Clinical] (zeros if missing) -> inner_dim;
            # "sum" -> mean of the available core modalities
            if self.fusion_type == "sum":
                x = torch.stack(list(core.values()), dim=0).mean(dim=0) if core else missing_embedding()
            elif self.core_modalities:
                x = self.core_compress(torch.cat([core.get(m, missing_embedding()) for m in self.core_modalities], dim=1))
            else:
                x = missing_embedding()
            core_vector = x
        if self.uses_hub:
            # radiology hub: core query, pooled CT / MRI tokens as context (same LN / FF path without radiology)
            x, hub_attention = self.radiology_hub(x, radiology_tokens)
            if hub_attention is not None:
                XA_attentions["att_core_to_radiology"] = hub_attention.detach()
            if self.training and self.multilevel_loss and radiology_tokens:
                # every smaller radiology configuration of the patient: logits for the multi-level loss
                names = sorted(radiology_tokens)
                sub_logits = {}
                for r in range(len(names)):
                    for subset in itertools.combinations(names, r):
                        h, _ = self.radiology_hub(core_vector, {m: radiology_tokens[m] for m in subset})
                        sub_logits[frozenset(subset)] = self.output_layer(self.dropout(h)) + late_clinical
                multilevel = (names, sub_logits)
        # fused / per-modality representations, for the shortcut probes (utils/shortcut_analysis.py)
        embeddings = {"fused": x.detach(), "core": core_vector.detach()}   # fused = input of the head
        if wsi_on:
            embeddings["WSI"] = wsi_embedding.detach()
        if genomics_on:
            embeddings["Genomics"] = genomics_embedding.detach()
        for m, t in radiology_tokens.items():
            embeddings[m] = t.mean(dim=1).detach()                          # mean of the pooled tokens
        if clinical_on:
            embeddings["Clinical"] = clinical_embedding.detach()
        # input-level representations (before any learned interaction): what the probes find here is in the
        # data itself (e.g. site signatures of the frozen WSI features, RNA batch effects), not learned
        if wsi_on:
            embeddings["WSI_input"] = data['patch_features'][~data['mask'].bool()].mean(dim=0, keepdim=True).detach()
        if genomics_on:
            embeddings["Genomics_input"] = torch.cat([data["genomics"][k] for k in self.genomics_group_name], dim=-1).detach()
        for m, raw in radiology_raw.items():
            embeddings[f"{m}_input"] = raw.mean(dim=1).detach()             # = the mean-pooled exam vector
        if clinical_on:
            embeddings["Clinical_input"] = clinical_tokens.mul(clinical_token_mask.unsqueeze(-1)).sum(dim=1).div(
                clinical_token_mask.sum(dim=1, keepdim=True).clamp(min=1)).detach()   # tokens before cross-attention

        # module B: adversary on the representation used by the head (training only)
        if self.training and getattr(self, "shortcut_adversary_weight", 0) > 0:
            on = {"WSI": wsi_on, "Genomics": genomics_on, "CT": ct_on, "MRI": mri_on, "Clinical": clinical_on}
            adversary_loss = self._adversary_loss(x, data, on)
            aux_loss = adversary_loss if aux_loss is None else aux_loss + adversary_loss

        # Output layer
        x = self.dropout(x)
        logits = self.output_layer(x) + late_clinical  # Shape: (batch_size, output_dim)

        output = {'output': logits, 'XA_attentions': XA_attentions, 'embeddings': embeddings}
        if multilevel is not None:
            # multi-level loss (computed by ModelManager, which has the labels): the survival loss on every smaller
            # configuration, and KL(parent || child) for every configuration and each parent with one exam less
            names, sub_logits = multilevel
            all_logits = {**sub_logits, frozenset(names): logits}
            pairs = [(all_logits[c], all_logits[c - {m}]) for c in all_logits for m in c]
            output['multilevel'] = {"subsets": list(sub_logits.values()), "pairs": pairs,
                                    "aux_weight": self.multilevel_aux_weight,
                                    "consistency_weight": self.multilevel_consistency_weight}
        if wsi_on:
            output['attention'] = A_out
        if aux_loss is not None:
            output['aux_loss'] = aux_loss
        if risk_std is not None:
            output['risk_std'] = risk_std
            output['posterior'] = (posterior_mu.detach(), posterior_logvar.detach())
        return output


class OXA_MISS_final_xa_graph_clinical_fusion(OXA_MISS_final):
    """Nome atteso da main.py (model.name = nome del file = nome della classe)."""


# ===========================================================================
# Test su dati sintetici (WSI/Genomica/CT/MRI) + clinica reale (dataloader/clinical_tokens.py)
# ===========================================================================
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dataloader.clinical_tokens import ClinicalTokens
    torch.manual_seed(0)
    store = ClinicalTokens()
    cid = next(iter(store.row))
    mean, std = store.num_stats(list(store.row))   # solo per il test: in CV sui pazienti di training del fold
    gname = ["tumor_suppression", "oncogenesis", "protein_kinases", "cellular_differentiation", "cytokines_and_growth"]
    gdim = [82, 313, 496, 331, 427]
    ALL = ["WSI", "Genomics", "Clinical", "CT", "MRI"]

    def make(case_id, wsi=True, gen=True, ct=True, mri=True):
        d = {"WSI_status": torch.tensor([wsi]), "patch_features": torch.randn(1, 30, 1024), "mask": torch.zeros(1, 30),
             "genomics_status": torch.tensor([gen]), "genomics": {k: torch.randn(1, n) for k, n in zip(gname, gdim)},
             "ct_status": torch.tensor([ct]), "ct_features": torch.randn(1, 20, 768),     # 20 CT region tokens
             "mri_status": torch.tensor([mri]), "mri_features": torch.randn(1, 12, 320)}  # 12 MRI region tokens
        c = store.get(case_id) if store.has(case_id) else store.missing()
        d.update({k: (torch.tensor([v]) if isinstance(v, bool) else v.unsqueeze(0)) for k, v in c.items()})
        return d

    modes = store.cat_modes(list(store.row))

    def build(**kw):
        kw.setdefault("radiology_dropout", 0.0)   # deterministic gradient checks
        m = OXA_MISS_final(input_dim=1024, inner_dim=64, ct_emb_dim=768, mri_emb_dim=320, **kw)
        if "Clinical" in m.input_modalities:
            m.set_clinical_normalization(mean, std, cat_mode=modes)
        return m

    # ogni flag di ablation, uno alla volta: forward + backward, gradiente su ogni parametro usabile
    flags = {"wsi_gated_keys": [False], "genomics_single_token": [True], "xa_shared_blocks": [True],
             "xa_update": ["sum"], "clinical_missing": ["impute", "indicator"], "clinical_xa": [False]}
    for name, values in flags.items():
        for value in values:
            m = build(input_modalities=ALL, clinical_token_dropout=0.0, **{name: value}); m.train()
            out = m(make(cid)); out["output"].sum().backward()
            no_grad = sorted(n for n, p in m.named_parameters()
                             if p.grad is None and not n.startswith("clinical_tokenizer.cat_embeddings"))
            print(f"flag {name}={value!s:9} output={tuple(out['output'].shape)} parametri senza gradiente: {no_grad or 'nessuno'}")

    for fusion in ("concatenate", "sum"):
        for xa in (False, True):
            for clinical_xa in (False, True):
                m = build(fusion_type=fusion, xa_fusion=xa, clinical_xa=clinical_xa, input_modalities=ALL,
                          clinical_token_dropout=0.0)   # senza dropout dei token: ogni parametro usato riceve gradiente
                m.train()
                out = m(make(cid))
                out["output"].sum().backward()
                # parametri SENZA gradiente, al netto di quelli che il dato non puo' toccare (categorie non osservate)
                no_grad = sorted(n for n, p in m.named_parameters()
                                 if p.grad is None and not n.startswith("clinical_tokenizer.cat_embeddings"))
                expected = [] if xa else sorted(n for n, _ in m.named_parameters() if n.startswith("xa_"))
                status = "OK" if no_grad == expected else f"INATTESI {sorted(set(no_grad) ^ set(expected))}"
                print(f"fusion={fusion:11s} xa_fusion={xa!s:5} clinical_xa={clinical_xa!s:5} output={tuple(out['output'].shape)} "
                      f"XA keys={sorted(out['XA_attentions'])} | gradiente {status}")

    # module A: product of experts
    m = build(input_modalities=ALL, fusion_type="poe", clinical_token_dropout=0.0, poe_subset_dropout=0.0); m.train()
    out = m(make(cid)); (out["output"].sum() + out["aux_loss"]).backward()
    no_grad = sorted(n for n, p in m.named_parameters() if p.grad is None and not n.startswith("clinical_tokenizer.cat_embeddings"))
    print(f"poe train: aux_loss={out['aux_loss'].item():.3f} parametri senza gradiente: {no_grad or 'nessuno'}")
    m.eval()
    with torch.no_grad():
        for kw in [dict(), dict(ct=False, mri=False), dict(gen=False, ct=False, mri=False)]:
            d = make(cid, **kw); r = m(d)
            print(f"poe eval {sorted(r['embeddings'])[:-1] if False else [k for k in r['embeddings'] if k != 'fused']}: "
                  f"risk_std={r['risk_std'].item():.4f} posterior var mean={r['posterior'][1].exp().mean().item():.4f}")
        voa = m.value_of_acquisition(make(cid, gen=False, ct=False, mri=False))
        print("value of acquisition (patient con WSI + Clinical):", {k: round(v.item(), 6) for k, v in voa["value"].items()},
              "| risk var", round(voa["risk_var"].item(), 6))

    # module B: adversary (gradient reversal), conditional on cancer type
    for fusion in ("sum", "poe"):
        m = build(input_modalities=ALL, fusion_type=fusion, shortcut_adversary_weight=0.1, shortcut_adversary_target="both",
                  n_studies=2, n_cancer_types=1, clinical_token_dropout=0.0, poe_subset_dropout=0.0); m.train()
        d = make(cid); d["study_index"] = torch.tensor([1]); d["cancer_type_index"] = torch.tensor([0])
        out = m(d); out["aux_loss"].backward()
        g_adv = m.shortcut_adversary[0].weight.grad.abs().sum().item()
        g_enc = m.inner_proj.weight.grad.abs().sum().item()
        print(f"adversary fusion={fusion}: aux_loss={out['aux_loss'].item():.3f} grad adversary {g_adv:.3e}, grad encoder {g_enc:.3e}")
    try:
        build(input_modalities=ALL, fusion_type="concatenate", shortcut_adversary_weight=0.1)
    except ValueError as e:
        print("adversary + concatenate rifiutato:", str(e)[:60])

    # radiology hub
    m = build(input_modalities=ALL); m.eval()
    with torch.no_grad():
        norms = {}
        hook = m.radiology_hub.norm2.register_forward_hook(lambda mod, i, o: norms.__setitem__("hub", o.norm(dim=-1).mean().item()))
        for ct, mri in [(True, True), (True, False), (False, False)]:
            r = m(make(cid, ct=ct, mri=mri))
            print(f"hub CT={ct!s:5} MRI={mri!s:5}: norma dell'uscita dell'hub {norms['hub']:.2f} | "
                  f"att_core_to_radiology {'presente' if 'att_core_to_radiology' in r['XA_attentions'] else 'assente'}")
        hook.remove()
    m.train()
    out = m(make(cid))
    ml = out["multilevel"]
    print(f"multilevel (CT+MRI): {len(ml['subsets'])} configurazioni minori (core, core+CT, core+MRI), {len(ml['pairs'])} coppie figlio/genitore")
    m.radiology_dropout = 0.3
    kept = sum("CT" in m(make(cid, mri=False))["embeddings"] for _ in range(300)) / 300
    print(f"radiology_dropout 0.3: CT tenuta nel {kept:.0%} dei forward di training")
    m = build(input_modalities=ALL, fusion_type="poe", poe_subset_dropout=0.0); m.eval()
    voa = m.value_of_acquisition(make(cid, gen=False, ct=False, mri=False))
    print(f"poe: esperti {m.poe_modalities}, hub {m.radiology_hub} | value of acquisition per {sorted(voa['value'])}")

    m = build(xa_fusion=False, clinical_xa=True, input_modalities=["WSI", "Genomics", "Clinical"])
    m.eval(); m.return_xa_attentions = True
    print("xa_fusion=False + return_xa_attentions ->", sorted(k for k, v in m(make(cid))["XA_attentions"].items() if v is not None))
    print("senza WSI       ->", tuple(m(make(cid, wsi=False))["output"].shape))
    print("senza clinica   ->", tuple(m(make("TCGA-XX-0000"))["output"].shape))

    # scala: la norma della rappresentazione aggiornata non dipende da quante modalita' ci sono
    m = build(input_modalities=["WSI", "Genomics", "Clinical"]); m.eval()
    with torch.no_grad():
        for combo in [(True, True), (True, False), (False, True)]:
            d = make(cid, gen=combo[0])
            if not combo[1]:
                d["clinical_status"] = torch.tensor([False])
            norms = {}
            hook = m.xa_norms["patches"].register_forward_hook(lambda mod, i, o: norms.__setitem__("patches", o.norm(dim=-1).mean().item()))
            m(d); hook.remove()
            print(f"genomica={combo[0]!s:5} clinica={combo[1]!s:5} -> norma media dei token WSI aggiornati {norms['patches']:.2f}")

    # maschera per paziente (batch di 2 nel tokenizer e nel dropout): ogni paziente tiene i propri token
    tok = m.clinical_tokenizer
    num, nmask = torch.tensor([[60.0], [0.0]]), torch.tensor([[True], [False]])
    cat = torch.tensor([[1, 2, 0, 0, 1], [2, 0, 3, 1, 0]])
    _, tmask = tok(num, cat, nmask)
    print("maschera per paziente:", tmask.int().tolist())
    m.train(); m.clinical_token_dropout = 0.9
    kept = torch.stack([m._drop_tokens(tmask) for _ in range(500)])
    print("dropout: mai fuori dalla maschera", bool((kept & ~tmask).sum() == 0), "| ogni paziente tiene >= 1 token",
          bool(kept.any(dim=2).all()))
    # checkpoint autosufficiente: media/std dell'eta' nello state_dict
    print("buffer nel checkpoint:", {k: [round(x, 2) for x in v.tolist()] for k, v in m.state_dict().items() if k.endswith(("num_mean", "num_std"))})
    try:
        OXA_MISS_final(input_modalities=["WSI", "CNV"])
    except ValueError as e:
        print("CNV rifiutato:", e)
