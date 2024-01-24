import math
import torch
import torch.nn as nn
class Temporal_MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(Temporal_MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Multi-head self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

    def forward(self, q,k,v):
        # triu函数将输入张量的下三角部分（包括主对角线）设置为0，只保留上三角部分。通过设置diagonal=1，表示保留主对角线上方的部分，将主对角线及其以下部分设置为0。
        # == 1：将上三角矩阵中的所有元素与1进行比较，生成一个二进制掩码张量。值为1的位置表示需要保留的位置，值为0的位置表示需要屏蔽的位置。
        Z_attention, Z_attention_w = self.self_attn(q, k, v)

        return Z_attention

if __name__ == '__main__':
    # Example usage:
    num_heads = 8

    batch_size = 32
    channel = 16
    seq_len = 15
    H = 48
    W = 48

    d_model = channel
    # K是字块大小
    Z = torch.randn(batch_size, seq_len, H, W, channel)
    temp_MHSA = Temporal_MultiHeadSelfAttention(d_model,num_heads)

    output = temp_MHSA(Z)
    print(output.shape)