import math

import torch
import torch.nn as nn


class LocalMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model, K):
        super(LocalMultiHeadSelfAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.K = K
        self.head_dim = d_model // num_heads

        # Multi-head self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

    def forward(self, q, k, v):
        Z=q
        N, T, H, W, d_model = Z.size()

        # 计算总的局部块数P
        P = (H // self.K) * (W // self.K)
        Z_unfolded = Z.unfold(2, self.K, self.K).unfold(3, self.K, self.K)  # (N, T, sqrtP,sqrtp, K, K, dmodel)
        Z_patches = Z_unfolded.contiguous().view(N, T, P, self.K, self.K, d_model)
        # Reshape and evenly divide the input into P local patches
        N, T, P, H_p, W_p, d_model = Z_patches.size()
        Z_patches_reshaped = Z_patches.view(N * T * P, self.K * self.K, d_model).transpose(0, 1)

        # q,k,v
        q = Z_patches_reshaped
        k = k.view(N * T * P, self.K * self.K, d_model).transpose(0, 1)
        v = v.view(N * T * P, self.K * self.K, d_model).transpose(0, 1)
        Z_patches_attention, Z_patches_attention_w = self.self_attn(q, k, v)
        Z_patches_attention = Z_patches_attention.transpose(0, 1).view(N, T, P, H_p, W_p, d_model)
        P_temp = int(math.sqrt(P))
        #拼接
        temp_attention = Z_patches_attention.reshape(N, T, P_temp, P_temp, H_p, W_p, d_model)#.permute(0, 1, 4, 2, 5,3,6)
        Z_attention = temp_attention.contiguous().view(N, T, H_p*P_temp, W_p*P_temp, d_model)
        return Z_attention




