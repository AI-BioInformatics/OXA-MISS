# File: adapted_prosurv.py

import torch
# import torch.nn as nn
from torch import nn, einsum
from math import ceil
import torch.nn.functional as F

from einops import rearrange, reduce


from collections import OrderedDict
from os.path import join
import pdb, math
# import eniops

import numpy as np





def get_sim_loss(similarity, label, censor):
    similarity_positive_mean = []
    similarity_negative_mean = []
    for i in range(label.shape[0]):
        if censor[i] == 0:
            mask = torch.zeros_like(similarity[i], dtype=torch.bool) # torch.Size([2, 32])
            mask[label[i].item(), :] = True
            similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
            similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
            similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
            similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

        else:
            if label[i] == 0:
                similarity_positive_mean.append(torch.mean(torch.mean(similarity[i], dim=-1), dim=-1)) # tensor
                # similarity_negative_mean.append(torch.tensor(0, dtype=torch.float).cuda())
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # print('device: ', device)
                similarity_negative_mean.append(torch.tensor(0, dtype=torch.float).to(device))
            else:   
                mask = torch.zeros_like(similarity[i], dtype=torch.bool)
                mask[label[i].item():, :] = True
                similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
                similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
                similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
                similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

    # 将列表转换为张量并求和
    similarity_positive_mean = torch.stack(similarity_positive_mean) #[B]
    similarity_negative_mean = torch.stack(similarity_negative_mean) #[B]

    positive_mean_sum = torch.sum(similarity_positive_mean)
    negative_mean_sum = torch.sum(similarity_negative_mean)

    sim_loss = -positive_mean_sum + negative_mean_sum

    return sim_loss

def get_align_loss(read_feat, original_feat, align_fn='mse', reduction='none'):
    if align_fn == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif align_fn == 'l1':
        loss_fn = nn.L1Loss(reduction=reduction)
    else:
        raise NotImplementedError
    
    return torch.sum(torch.mean(loss_fn(read_feat, original_feat.detach()), dim=-1), dim=-1)



class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            # add
            nn.LayerNorm(dim2),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class MultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim = 256,
                 k_dim = 256,
                 v_dim = 256,
                 embed_dim = 256,
                 out_dim = 256,
                 n_head = 4,
                 dropout=0.1,
                 temperature = 1
                 ):
        super(MultiheadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = self.embed_dim//self.n_head
        self.temperature = temperature


        self.w_q = nn.Linear(self.q_dim, embed_dim)
        self.w_k = nn.Linear(self.k_dim, embed_dim)
        self.w_v = nn.Linear(self.v_dim, embed_dim)

        self.scale = (self.embed_dim//self.n_head) ** -0.5

        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        # self.layerNorm1 = nn.LayerNorm(out_dim)
        # self.layerNorm2 = nn.LayerNorm(out_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        
        # self.feedForward = nn.Sequential(
        #     nn.Linear(out_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, out_dim)
        # )

    def forward(self, q, k, v, return_attn = False):
        q_raw = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = q.shape[0] # B
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_score = F.softmax(attention_score / self.temperature, dim = -1)

        attention_score = self.attn_dropout(attention_score)

        x = torch.matmul(attention_score, v)

        attention_score = attention_score.sum(dim = 1)/self.n_head
        
        attn_out = x.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_out = self.out_proj(attn_out)

        attn_out = self.proj_dropout(attn_out)

        # attn_out = attn_out + q_raw

        # attn_out = self.layerNorm1(attn_out)

        # out = self.feedForward(attn_out)

        # out = self.layerNorm2(out + attn_out)

        # out = self.dropout(out)
        if return_attn:
            return attn_out, attention_score
        else:
            return attn_out
        # return out, attention_score


class GAN(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hiden_dim=512):
        super(GAN, self).__init__()
        self.generator = Generator(input_dim, output_dim, hiden_dim)
        self.discriminator = Discriminator(input_dim, hiden_dim)
    def forward(self, x):
        return self.generator(x), self.discriminator(x)

# 生成器
class Generator(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hiden_dim=512):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hiden_dim),
            nn.ReLU(),
            nn.Linear(hiden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.fc(x)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hiden_dim=512):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hiden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hiden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
        

class VAE_Encoder(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.fc3_mean = nn.Linear(input_dim//4, input_dim//8)
        self.fc3_logvar = nn.Linear(input_dim//4, input_dim//8)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

# 解码器
class VAE_Decoder(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim//8, input_dim//4)
        self.fc2 = nn.Linear(input_dim//4, input_dim//2)
        self.fc3 = nn.Linear(input_dim//2, input_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

# VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim)
        self.decoder = VAE_Decoder(input_dim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

def vae_loss(reconstructed_x, x, mean, logvar):
    mse_loss = nn.MSELosss(reconstructed_x, x.detach(), reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return mse_loss + kld_loss


def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            # add
            nn.LayerNorm(dim2),
            nn.AlphaDropout(p=dropout, inplace=False))

class pathMIL(nn.Module):
    def __init__(self, model_type = 'TransMIL', input_dim = 256, dropout=0.1):
        super(pathMIL, self).__init__()

        self.model_type = model_type

        if model_type == 'TransMIL':
            self.translayer1 = TransLayer(dim = input_dim)
            self.translayer2 = TransLayer(dim = input_dim)
            self.pos_layer = PPEG(dim = input_dim)
        elif model_type == 'ABMIL':
            self.path_gated_attn = Attn_Net_Gated(L=input_dim, D=input_dim, dropout=dropout, n_classes=1)

    def forward(self, h_path):

        if self.model_type == 'TransMIL':
            H = h_path.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h_path_sa = torch.cat([h_path, h_path[:,:add_length,:]], dim = 1) #[B, N, 512]
            # cls_token
            # B = h_path_sa.shape[0]
            # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            # h_path_sa = torch.cat((cls_tokens, h_path_sa), dim=1)
            # Translayer1
            h_path_sa = self.translayer1(h_path_sa) #[B, N, 256]
            # PPEG
            h_path_sa = self.pos_layer(h_path_sa, _H, _W) #[B, N, 256]
            # Translayer2
            h_path_sa = self.translayer2(h_path_sa) #[B, N, 256]
            # cls_token
            # h_path_sa = self.norm(h_path_sa)[:,0]
            h_path_sa = torch.mean(h_path, dim=1) #[B, 256]

            return h_path_sa

        elif self.model_type == 'ABMIL':
            A, h_path = self.path_gated_attn(h_path)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=-1) 
            h_path = torch.matmul(A, h_path).squeeze(1) #[B, D]
            return h_path
        
        else:
            raise NotImplementedError
            return 


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//4,
            heads = 4,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
    

class ProSurvOriginal(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 fusion='concat', 
                 n_classes=4,
                 model_size_path: str='small', 
                 model_size_geno: str='small', 
                 mil_model_type='TransMIL',
                 geno_mlp_type='SNN',
                 memory_size=32,
                 dropout=0.25):
        
        super(ProSurvOriginal, self).__init__()
        self.fusion = fusion
        self.geno_input_dim = omic_input_dim
        self.n_classes = n_classes
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}

        ### Define Prototype Bank
        self.memory_size = memory_size
        self.memory_dim = 256

        self.path_prototype_bank = nn.Parameter(torch.empty(self.n_classes, self.memory_size, self.memory_dim))
        self.geno_prototype_bank = nn.Parameter(torch.empty(self.n_classes, self.memory_size, self.memory_dim))
        torch.nn.init.xavier_uniform_(self.path_prototype_bank, gain=1.0)
        torch.nn.init.xavier_uniform_(self.geno_prototype_bank, gain=1.0)

        ### pathlogy FC
        size = self.size_dict_path[model_size_path]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.LayerNorm(normalized_shape = size[1]))
        fc.append(nn.Dropout(dropout))
        self.path_proj = nn.Sequential(*fc)

        self.path_attn_net = pathMIL(model_type=mil_model_type, input_dim=size[1], dropout=dropout)
        
        ### Genomic SNN
        hidden = self.size_dict_geno[model_size_geno]
        if geno_mlp_type == 'SNN':
            geno_snn = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                geno_snn.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
            self.geno_snn = nn.Sequential(*geno_snn)
        else:
            self.geno_snn = nn.Sequential(
                nn.Linear(omic_input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout))

        ### Multihead Attention
        self.path_intra_read_attn = MultiheadAttention(q_dim = self.size_dict_geno[model_size_geno][-1], k_dim = self.memory_dim, 
                                        v_dim = self.memory_dim, embed_dim = size[1], out_dim = size[1], 
                                        n_head = 4, dropout=dropout, temperature=0.5)

        self.geno_intra_read_attn = MultiheadAttention(q_dim = size[1], k_dim = self.memory_dim, 
                                        v_dim = self.memory_dim, embed_dim = size[1], out_dim = size[1], 
                                        n_head = 4, dropout=dropout, temperature=0.5)
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(size[1]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU()])
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
    def forward(self, **kwargs):
        # input data
        x_path = kwargs['x_path']
        x_geno = kwargs['x_omic']
        label = kwargs['label']
        censor = kwargs['censor']
        is_training = kwargs['training']
        input_modality = kwargs['input_modality']

        if x_path!=None:
            batch_size = x_path.shape[0]
        elif x_geno!=None:
            batch_size = x_geno.shape[0]
        else:
            raise NotImplementedError

        # pathlogy projection
        if x_path!=None:
            h_path = self.path_proj(x_path) #[B, n_patchs, D]
            # pathlogy attention net
            h_path = self.path_attn_net(h_path) #[B, D]

        # Genomic SNN
        if x_geno!=None:
            h_geno = self.geno_snn(x_geno).squeeze(1) #[B, D]

        if is_training:
            # 先归一化，再计算相似度
            # similarity
            path_sim_loss = 0.
            geno_sim_loss = 0.

            if input_modality in ['path', 'path_and_geno']:
                path_prototype_norm = F.normalize(self.path_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                h_path_norm = F.normalize(h_path) #[B, D]
                path_similarity = torch.matmul(h_path_norm, torch.transpose(path_prototype_norm, 0, 1)).reshape(
                    -1, self.n_classes, self.memory_size) #[B, n_classes, size]
                
                path_sim_loss = get_sim_loss(path_similarity, label, censor)

            if input_modality in ['geno', 'path_and_geno']:
                geno_prototype_norm = F.normalize(self.geno_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                h_geno_norm = F.normalize(h_geno) #[B, D]
                geno_similarity = torch.matmul(h_geno_norm, torch.transpose(geno_prototype_norm, 0, 1)).reshape(
                    -1, self.n_classes, self.memory_size) #[B, n_classes, size]
                geno_sim_loss = get_sim_loss(geno_similarity, label, censor)
            
            sim_loss = path_sim_loss + geno_sim_loss

        # intra-modal read attention
        if input_modality in ['geno', 'path_and_geno']:
            path_prototype_bank_flat = self.path_prototype_bank.reshape(
                self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
            h_path_read = self.path_intra_read_attn(h_geno.unsqueeze(1), path_prototype_bank_flat, path_prototype_bank_flat).squeeze(1) # [B, D]

        if input_modality in ['path', 'path_and_geno']:
            geno_prototype_bank_flat = self.geno_prototype_bank.reshape(
                self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
            h_geno_read = self.geno_intra_read_attn(h_path.unsqueeze(1), geno_prototype_bank_flat, geno_prototype_bank_flat).squeeze(1) # [B, D]

        if input_modality == 'path':
            h_path_read = h_path
            h_geno = h_geno_read
        elif input_modality == 'geno':
            h_geno_read = h_geno
            h_path = h_path_read
        elif input_modality == 'path_and_geno':
            pass
        else:
            raise NotImplementedError(f'input_modality: {input_modality} not suported')
                
        h_path_avg = (h_path + h_path_read) /2
        h_geno_avg = (h_geno + h_geno_read) /2

        if self.training:
            path_loss_align = 0.
            geno_loss_align = 0.
            
            if input_modality == 'path_and_geno':
                path_loss_align = get_align_loss(h_path_read, h_path)
                geno_loss_align = get_align_loss(h_geno_read, h_geno)

            loss_align = path_loss_align + geno_loss_align

        ### Fusion Layer
        if self.fusion == 'bilinear':
            h = self.mm(h_path_avg, h_geno_avg).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path_avg, h_geno_avg], dim=-1))
        else:
            h = self.mm(h_path)
                
        ### Survival Layer
        logits = self.classifier(h)
        
        if is_training:
            return logits, sim_loss, loss_align
        else:
            if kwargs['return_feature']:
                return logits, h_path, h_geno_read, h_geno, h_geno_read
            else:
                return logits


class ProSurv(nn.Module):
    """
    Un wrapper per il modello ProSurv per renderlo compatibile con la pipeline di dati
    progettata per Custom_Multimodal_XA.

    Questo modello accetta un dizionario 'data' nella sua funzione forward e gestisce
    le modalità WSI e Genomics. La modalità CNV viene ignorata.
    """
    def __init__(self,
                 # Parametri per la coerenza con la tua pipeline
                 n_classes=4,
                 genomics_group_name=["high_refractory", "high_sensitive", "hypoxia_pathway"],
                 genomics_group_input_dim=[25, 35, 31],
                 input_modalities = ["WSI", "Genomics"],
                 cnv_group_name= [],
                 
                 # Parametri specifici per ProSurv (possono essere esposti o hardcoded)
                 fusion='concat',
                 model_size_path: str='small',
                 model_size_geno: str='small',
                 mil_model_type='TransMIL',
                 geno_mlp_type='SNN',
                 memory_size=32,
                 dropout=0.25):
        
        super(ProSurv, self).__init__()
        
        self.genomics_group_name = genomics_group_name
        self.input_modalities = ["WSI", "Genomics"] # Questo modello gestisce solo WSI e Genomics

        # ProSurv si aspetta un unico input per i dati omici.
        # Calcoliamo la dimensione totale concatenando i gruppi genomici.
        total_omic_input_dim = sum(genomics_group_input_dim)

        # Istanzia il modello ProSurv originale
        self.prosurv_model = ProSurvOriginal(
            omic_input_dim=total_omic_input_dim,
            fusion=fusion,
            n_classes=n_classes,
            model_size_path=model_size_path,
            model_size_geno=model_size_geno,
            mil_model_type=mil_model_type,
            geno_mlp_type=geno_mlp_type,
            memory_size=memory_size,
            dropout=dropout
        )

    def forward(self, data):
        """
        Funzione forward che accetta un dizionario 'data' e lo adatta per ProSurv.

        Args:
            data (dict): Un dizionario contenente i dati. Deve avere le seguenti chiavi:
                'WSI_status' (bool): True se i dati WSI sono presenti.
                'genomics_status' (bool): True se i dati genomici sono presenti.
                'patch_features' (torch.Tensor): Feature delle patch WSI.
                'mask' (torch.Tensor): Maschera per le patch.
                'genomics' (dict): Dizionario con i dati dei gruppi genomici.
                'label' (torch.Tensor): Etichette per il calcolo della loss (necessario per ProSurv).
                'censor' (torch.Tensor): Dati di censura (necessario per ProSurv).
        
        Returns:
            dict: Un dizionario contenente:
                'output' (torch.Tensor): I logits del classificatore.
                'sim_loss' (torch.Tensor): Loss di similarità (solo in training).
                'loss_align' (torch.Tensor): Loss di allineamento (solo in training).
        """
        x_path, x_omic = None, None
        
        # --- 1. Preparazione dei dati WSI ---
        wsi_present = "WSI" in self.input_modalities and data.get("WSI_status", torch.tensor(False)).item()
        if wsi_present:
            patch_embeddings = data['patch_features']
            mask = data['mask']
            # Applica la maschera e aggiunge la dimensione del batch, come nel tuo modello originale
            x_path = patch_embeddings[~mask.bool()].unsqueeze(0)

        # --- 2. Preparazione dei dati Genomici ---
        genomics_present = "Genomics" in self.input_modalities and data.get("genomics_status", torch.tensor(False)).item()
        if genomics_present:
            genomics_data = data["genomics"]
            # Concatena i gruppi genomici per creare un singolo tensore per ProSurv
            genomics_groups = []
            for key in self.genomics_group_name:
                genomics_groups.append(genomics_data[key])
            x_omic = torch.cat(genomics_groups, dim=-1)

        # --- 3. Preparazione degli altri argomenti per ProSurv ---
        if wsi_present and genomics_present:
            input_modality = 'path_and_geno'
        elif wsi_present:
            input_modality = 'path'
        elif genomics_present:
            input_modality = 'geno'
        else:
            raise ValueError("Almeno una modalità (WSI o Genomics) deve essere fornita.")

        # ProSurv richiede 'label' e 'censor' direttamente nella forward per calcolare le loss interne
        label = data['label']
        censor = data['censorship']
        
        # --- 4. Chiamata al modello ProSurv ---
        prosurv_args = {
            'x_path': x_path, # torch.Size([1, 4096, 1024])
            'x_omic': x_omic, # torch.Size([1, 1, 1556]) -> in realta qui è torch.Size([1, 1649])
            'label': label, # torch.Size([1])
            'censor': censor, # torch.Size([1])
            'training': self.training, # Passa lo stato corrente del modello (train/eval)
            'input_modality': input_modality,
            'return_feature': False # Non ci servono le feature intermedie
        }
        
        if self.training:
            logits, sim_loss, loss_align = self.prosurv_model(**prosurv_args)
            # output = {
            #     'output': logits,
            #     'sim_loss': sim_loss,
            #     'loss_align': loss_align
            # }
            output = {
                'output': logits,
                'partial_loss': sim_loss*0.2 + loss_align*0.2
            }
        else:
            logits = self.prosurv_model(**prosurv_args)
            output = {'output': logits}

        return output