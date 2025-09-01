import torch 
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self,input_dim=1024,inner_dim=128, output_dim=4):
        super(ABMIL,self).__init__()
        self.inner_proj = nn.Linear(input_dim, inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

        # Output layer
        self.output_layer = nn.Linear(inner_dim, output_dim)
        
    def forward(self, x):
        # Extract patch features
        # x = x['patch_features']  # x is a dictionary with key 'patch_features'
        
        # mettere proiezione iniziale
        x = self.inner_proj(x)

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
        # output = {'output': output}
        return output
    

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, normalize_emb, aggr="mean", **kwargs):
        super(EdgeSAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.normalize_emb = normalize_emb

        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_activation = nn.ReLU()
        self.update_activation = nn.ReLU()

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index):
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, edge_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels,
                                                     self.edge_channels)


class GNNStack(torch.nn.Module):
    def __init__(self, node_channels, edge_channels, normalize_embs, num_layers, dropout):
        super(GNNStack, self).__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.normalize_embs = normalize_embs
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = self.build_convs(node_channels, edge_channels, normalize_embs, num_layers)
        self.edge_update_mlps = self.build_edge_update_mlps(node_channels, edge_channels, num_layers)

    def build_convs(self, node_channels, edge_channels, normalize_embs, num_layers):
        convs = nn.ModuleList()
        for l in range(num_layers):
            conv = EdgeSAGEConv(node_channels, node_channels, edge_channels, normalize_embs[l])
            convs.append(conv)
        return convs

    def build_edge_update_mlps(self, node_channels, edge_channels, num_layers):
        edge_update_mlps = nn.ModuleList()
        for l in range(num_layers - 1):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_channels + node_channels + edge_channels, edge_channels),
                nn.ReLU()
            )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_attr, edge_index)
            if l < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        return x


class MML(nn.Module):
    def __init__(self, num_modalities, hidden_channels, normalize_embs, num_layers, dropout, num_classes):
        super(MML, self).__init__()
        self.num_classes = num_classes
        self.modality_nodes = nn.Parameter(torch.randn(num_modalities, hidden_channels))
        if normalize_embs is None:
            normalize_embs = [False] * num_layers
        else:
            normalize_embs = [bool(int(i)) for i in normalize_embs]
        self.gnn = GNNStack(hidden_channels, hidden_channels, normalize_embs, num_layers, dropout)
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)

        self.cl_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        if num_classes == 1:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softmax(dim=-1)

    def edgedrop(self, flag):
        n, m = flag.size()
        for i in range(n):
            count_ones = flag[i].sum().item()

            if torch.rand(1) < 0.5:
                continue

            if count_ones <= 1:
                continue

            # Randomly choose how many 1s we want to keep
            keep_count = torch.randint(1, count_ones, (1,)).item()

            # Get the indices of 1s in the row
            one_indices = (flag[i] == 1).nonzero(as_tuple=True)[0]

            # Randomly shuffle the indices
            one_indices = one_indices[torch.randperm(one_indices.size(0))]

            # Set the 1s we don't want to keep to 0
            mask_count = count_ones - keep_count
            flag[i][one_indices[:mask_count]] = 0

        return flag

    def unsup_ce_loss(self, zaz_s):
        target = torch.arange(zaz_s.size(0), device=zaz_s.device)
        loss = F.cross_entropy(zaz_s, target)
        loss_t = F.cross_entropy(zaz_s.t(), target)
        return (loss + loss_t) / 2

    def sup_ce_loss(self, z_s, zaz_s, y):
        y = y.view(-1, 1)
        target = (y == y.t()).float()
        loss_z = F.binary_cross_entropy_with_logits(z_s, target)
        loss_zaz = F.binary_cross_entropy_with_logits(zaz_s, target)
        loss_zaz_t = F.binary_cross_entropy_with_logits(zaz_s.t(), target)
        return (2 * loss_z + loss_zaz + loss_zaz_t) / 4

    def classification_loss(self, l, y):
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(l.squeeze(-1), y)
        else:
            loss = F.cross_entropy(l, y)
        return loss

    def forward(self,
                x1, x1_flag,
                x2, x2_flag,
                y, y_flag=True):
        batch_size = x1.size(0)
        hidden_dim = x1.size(1)

        x_flag = torch.stack([x1_flag, x2_flag], dim=1)
        x = torch.stack([x1, x2], dim=1)

        g_patient_nodes = torch.ones(batch_size, hidden_dim)
        g_patient_nodes = g_patient_nodes.to(x1.device)
        g_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        g_edge_index = x_flag.nonzero().t()
        g_edge_index[1] += batch_size
        g_edge_index = torch.cat([g_edge_index, g_edge_index.flip([0])], dim=1)
        g_edge_attr = x[x_flag].repeat(2, 1)
        z = self.gnn(g_nodes, g_edge_attr, g_edge_index)

        ag_x_flag = x_flag.clone()
        ag_x_flag = self.edgedrop(ag_x_flag)
        ag_patient_nodes = torch.ones(batch_size, hidden_dim)
        ag_patient_nodes = ag_patient_nodes.to(x1.device)
        ag_nodes = torch.cat([ag_patient_nodes, self.modality_nodes], dim=0)
        ag_edge_index = ag_x_flag.nonzero().t()
        ag_edge_index[1] += batch_size
        ag_edge_index = torch.cat([ag_edge_index, ag_edge_index.flip([0])], dim=1)
        ag_edge_attr = x[ag_x_flag].repeat(2, 1)
        az = self.gnn(ag_nodes, ag_edge_attr, ag_edge_index)

        z = z[:batch_size]
        az = az[:batch_size]

        # contrastive learning loss
        u = self.cl_projection(z)
        u = F.normalize(u, dim=-1)
        au = self.cl_projection(az)
        au = F.normalize(au, dim=-1)
        uau_s = torch.matmul(u, au.t()) * self.tau
        # loss non supervisionata
        unsup_loss = self.unsup_ce_loss(uau_s)

        # suppose to have all the y
        # u = u[y_flag]
        # au = au[y_flag]
        # y = y[y_flag]
        u_s = torch.matmul(u, u.t()) * self.tau
        uau_s = torch.matmul(u, au.t()) * self.tau
        # loss supervisionata
        sup_loss = self.sup_ce_loss(u_s, uau_s, y)

        # suppose to have all the y
        # cls
        # z = z[y_flag]
        logits = self.classifier(z)
        # cls_loss = self.classification_loss(logits, y)

        return {'partial_loss': 0.5 * unsup_loss + 0.5 * sup_loss, 'output': logits }# + cls_loss

    def inference(self,
                  x1, x1_flag,
                  x2, x2_flag,):
        batch_size = x1.size(0)
        hidden_dim = x1.size(1)

        x_flag = torch.stack([x1_flag, x2_flag], dim=1)
        x = torch.stack([x1, x2], dim=1)

        g_patient_nodes = torch.ones(batch_size, hidden_dim)
        g_patient_nodes = g_patient_nodes.to(x1.device)
        g_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        g_edge_index = x_flag.nonzero().t()
        g_edge_index[1] += batch_size
        g_edge_index = torch.cat([g_edge_index, g_edge_index.flip([0])], dim=1)
        g_edge_attr = x[x_flag].repeat(2, 1)

        z = self.gnn(g_nodes, g_edge_attr, g_edge_index)
        z = z[:batch_size]

        logits = self.classifier(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        y_scores = self.act(logits)
        return y_scores, logits



####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


import torch
import torch.nn as nn


class MUSE(nn.Module):
    def __init__(
            self,
            embedding_size=64,
            dropout=0.25,
            ffn_layers=2,
            gnn_layers=2,
            output_size=4,
            gnn_norm=None,
            device="cuda",

            genomics_group_name=[],
            genomics_group_input_dim=[],
            genomics_group_dropout=[],

            input_modalities = ["WSI", "Genomics"],
            cnv_group_name = ["high_refractory", "high_sensitive",  "hypoxia_pathway"]
    ):
        super(MUSE, self).__init__()

        if len(genomics_group_dropout) != len(genomics_group_name):
            if len(genomics_group_dropout) == 1:
                genomics_group_dropout = genomics_group_dropout * len(genomics_group_name)
            else:
                raise ValueError(f"Mismatch between genomics_group_name ({len(genomics_group_name)}) and genomics_group_dropout lengths ({len(genomics_group_dropout)})")

        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.genomics_group_name = genomics_group_name
        self.genomics_group_input_dim = genomics_group_input_dim
        self.genomics_group_dropout = genomics_group_dropout

        self.dropout_layer = nn.Dropout(dropout)

        self.x1_encoder = ABMIL(output_dim=embedding_size)
        # self.x1_mapper = nn.Linear(embedding_size, embedding_size)

        # self.x2_encoder = FFNEncoder(input_dim=227,
        #                              hidden_dim=embedding_size,
        #                              output_dim=embedding_size,
        #                              num_layers=ffn_layers,
        #                              dropout_prob=dropout,
        #                              device=device)

        # self.genomic_encoder = {}
        self.x2_encoder = {}
        for name, input_dim, rate in zip(genomics_group_name, genomics_group_input_dim, genomics_group_dropout):
            self.x2_encoder[name] = nn.Sequential(
                                            nn.Dropout(rate),
                                            nn.Linear(input_dim, embedding_size),
                                            nn.ReLU(),
                                            nn.Linear(embedding_size, embedding_size),
                                        ) 
        self.x2_encoder = nn.ModuleDict(self.x2_encoder)

        # self.x2_mapper = nn.Linear(embedding_size, embedding_size)


        self.mml = MML(num_modalities=2,
                       hidden_channels=embedding_size,
                       num_layers=gnn_layers,
                       dropout=dropout,
                       normalize_embs=gnn_norm,
                       num_classes=output_size)

    def forward(
            self,
            # x1,
            # x1_flag,
            # x2,
            # x2_flag,
            data,
            label,
            # **kwargs,
    ):
        x1 = data['patch_features']
        x1_flag = data["WSI_status"]

        x2 = data["genomics"]
        x2_flag = data["genomics_status"]

        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)
        
        label = label.to(self.device)


        if x1_flag:
            x1_embedding = self.x1_encoder(x1)
            # x1_embedding = self.x1_mapper(x1_embedding)
            # x1_embedding[x1_flag == 0] = 0
            x1_embedding = self.dropout_layer(x1_embedding)
            

        if x2_flag:
            genomics = x2
            genomics_groups = []
            for key in self.genomics_group_name:
                genomics_group_i = genomics[key]
                genomics_group_i = self.x2_encoder[key](genomics_group_i)
                genomics_groups.append(genomics_group_i)           
            genomics_embedding = torch.stack(genomics_groups, dim=1)
            x2_embedding = genomics_embedding.sum(dim=1, keepdim=False) # self.x2_encoder(x2)
            # x2_embedding = self.dropout_layer(x2_embedding)

        if not x1_flag:
            x1_embedding = torch.zeros_like(x2_embedding)
        if not x2_flag:
            x2_embedding = torch.zeros_like(x1_embedding)

        # gnn
        # loss = self.mml(
        output = self.mml(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            label,
        )
        
        return output # --> {'partial_loss': .. , 'output': .. }

    def inference(
            self,
            data,
            **kwargs,
    ):
        x1 = data['patch_features']
        x1_flag = data["WSI_status"]

        x2 = data["genomics"]
        x2_flag = data["genomics_status"]

        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)

        if x1_flag:
            x1_embedding = self.x1_encoder(x1)
        # x1_embedding = self.x1_mapper(x1_embedding)

        if x2_flag:
            genomics = x2
            genomics_groups = []
            for key in self.genomics_group_name:
                genomics_group_i = genomics[key]
                genomics_group_i = self.x2_encoder[key](genomics_group_i)
                genomics_groups.append(genomics_group_i)           
            genomics_embedding = torch.stack(genomics_groups, dim=1)
            x2_embedding = genomics_embedding.sum(dim=1, keepdim=False)

        # x2_embedding = self.x2_mapper(x2_embedding)

        if not x1_flag:
            x1_embedding = torch.zeros_like(x2_embedding)
        if not x2_flag:
            x2_embedding = torch.zeros_like(x1_embedding)

        # gnn
        y_scores, logits = self.mml.inference(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
        )
        # return y_scores, logits
    
        return {'output': logits}
