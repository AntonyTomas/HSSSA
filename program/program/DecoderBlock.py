import torch
import torch.nn as nn
from mhd import LocalMultiHeadSelfAttention
from ConvFFN import ConvFFN
from temporal_MHSA import Temporal_MultiHeadSelfAttention
from DropPath import DropPath
from position_encoding import PositionEmbeddding1D,PositionEmbeddding2D

class DecoderBlock(nn.Module):
    def __init__(self, encH, encW, channels, num_heads, K=8, dropout=0., drop_path=0.,
                 Spatial_FFN_hidden_ratio=4, dim_feedforward=1024):
        super().__init__()
        # SLMHSA
        self.num_heads = num_heads
        embed_dim = channels
        self.embed_dim = embed_dim
        patch_size = K
        self.patch_size = patch_size
        self.norm1 = nn.LayerNorm(embed_dim)
        self.SL_MHSA = LocalMultiHeadSelfAttention(num_heads, embed_dim, patch_size)
        self.SL_POS = PositionEmbeddding2D()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio
        self.Conv_FFN = ConvFFN(
            encH,
            encW,
            in_features=embed_dim,
            hidden_features=int(Spatial_FFN_hidden_ratio * embed_dim),
            out_features=embed_dim,
            drop=dropout,
            AR_model=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = nn.LayerNorm(embed_dim)
        self.temporal_MHSA = Temporal_MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.TEM_POS = PositionEmbeddding1D()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop3 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm4 = nn.LayerNorm(embed_dim)

        # encoder-decoder attention, follow with conv feed-forward
        self.EncDecAttn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout)
        self.Conv_FFN2 = ConvFFN(
            encH,
            encW,
            in_features=embed_dim,
            hidden_features=int(Spatial_FFN_hidden_ratio * embed_dim),
            out_features=embed_dim,
            drop=dropout,
            AR_model=True)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.norm6 = nn.LayerNorm(embed_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, tgt, query_pos, memory):
        """
        tgt: (N, T2, H, W, C)
        query_pos: (N, T2, H, W, C)
        memory: (N, T1, H, W, C)


        Return: (N, T2, H, W, C)
        """
        N, T2, H, W, C = tgt.shape
        self.sl_posit = self.SL_POS(N=1, E=self.embed_dim, H=self.patch_size, W=self.patch_size)[0, ...].permute(1, 2,                                                                                               0)
        self.sl_pos = torch.tile(self.sl_posit, (H // self.patch_size, W // self.patch_size, 1))
        self.tem_pos = self.TEM_POS(L=T2, N=1, E=self.embed_dim)
        tgt2 = self.norm1(tgt)
        tgt2_query_pos = tgt2 + query_pos
        tgt2 = tgt + self.drop_path(self.SL_MHSA(q=tgt2_query_pos+self.sl_pos,k=tgt2_query_pos+self.sl_pos,v=tgt2))  # spatial local window self-attention, and skip connection
        # Conv feed-forward, different local window information interacts
        tgt2 = tgt2 + self.drop_path(self.Conv_FFN(self.norm2(tgt2)))  # (N, T, H, W, C)

        # query temporal self-attention
        tgt2 = tgt2.permute(1, 0, 2, 3, 4).reshape(T2, N * H * W, C)
        tgt = self.norm3(tgt2)
        tgt2 = tgt2 + self.drop1(self.temporal_MHSA(self.norm3(tgt2+self.tem_pos),self.norm3(tgt2+self.tem_pos),self.norm3(tgt2)))

        # feed-forward after temporal self-attention
        tgt = self.norm4(tgt2)
        tgt = self.linear2(self.drop2(self.activation(self.linear1(tgt))))
        tgt2 = tgt2 + self.drop3(tgt)

        tgt = self.norm5(tgt2)
        N1,T1,H1, W1, C1 = memory.shape
        self.past_tem_pos = self.TEM_POS(L=T1, N=1, E=C1)
        memory = memory.permute(1, 0, 2, 3, 4).reshape(T1, N * H * W, C)
        query_pos = query_pos.permute(1, 0, 2, 3, 4).reshape(T2, N * H * W, C)
        tgt2 = tgt2 + self.drop_path1(
            self.EncDecAttn(query=tgt + query_pos+self.tem_pos,
                            key=memory+self.past_tem_pos, value=memory)[0])
        tgt2 = tgt2.reshape(T2, N, H, W, C).permute(1, 0, 2, 3, 4)

        # another Conv feed-forward, different local window information interacts
        tgt2 = tgt2 + self.drop_path1(self.Conv_FFN(self.norm6(tgt2)))

        return tgt2

if __name__ == '__main__':
    from EncoderBlock import EncoderBlock
    device = torch.device("cuda:0")
    num_heads = 2
    batch_size = 2
    channel = 8
    seq_len = 15
    H = 48
    W = 48
    d_model = channel
    # K是字块大小
    Z = torch.randn(batch_size, seq_len, H, W, channel)
    Z = Z.to(device)
    encoderBlock1 = EncoderBlock(
        encH=H,
        encW=W,
        channels=channel,
        num_heads=num_heads,
        K=8,
        dropout=0.,
        drop_path=0.,
        Spatial_FFN_hidden_ratio=4,
        dim_feedforward=256,
    )
    encoderBlock2 = EncoderBlock(
        encH=H,
        encW=W,
        channels=channel,
        num_heads=num_heads,
        K=8,
        dropout=0.,
        drop_path=0.,
        Spatial_FFN_hidden_ratio=4,
        dim_feedforward=256,
    )
    encoderBlock1 = encoderBlock1.to(device)
    encoderBlock2 = encoderBlock2.to(device)
    memory1 = encoderBlock1(Z)
    memory2 = encoderBlock2(memory1)
    query_pos = nn.Parameter(torch.randn(batch_size, 7, H, W, channel))
    query_pos = query_pos.to(device)
    init_tgt = torch.zeros_like(query_pos, requires_grad=False)
    init_tgt = init_tgt.to(device)
    decoderBlock1 = DecoderBlock(
        encH = H,
        encW = W,
        channels = channel,
        num_heads = num_heads,
        K=8,
        dropout=0.,
        drop_path=0.,
        Spatial_FFN_hidden_ratio=4,
        dim_feedforward=256
    )
    decoderBlock2 = DecoderBlock(
        encH=H,
        encW=W,
        channels=channel,
        num_heads=num_heads,
        K=8,
        dropout=0.,
        drop_path=0.,
        Spatial_FFN_hidden_ratio=4,
        dim_feedforward=256
    )
    decoderBlock1 = decoderBlock1.to(device)
    decoderBlock2 = decoderBlock2.to(device)
    temp = decoderBlock2(init_tgt,query_pos,memory2)
    output = decoderBlock1(temp, query_pos, memory1)


    print(output.shape)
