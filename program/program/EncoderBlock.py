import torch
import torch.nn as nn
from mhd import LocalMultiHeadSelfAttention
from ConvFFN import ConvFFN
from temporal_MHSA import Temporal_MultiHeadSelfAttention
from DropPath import DropPath
from position_encoding import PositionEmbeddding1D,PositionEmbeddding2D

class EncoderBlock(nn.Module):
    def __init__(self, encH, encW, channels, num_heads, K=8, dropout=0., drop_path=0.,
                 Spatial_FFN_hidden_ratio=4, dim_feedforward=1024):
        super(EncoderBlock,self).__init__()
        #SLMHSA
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
            hidden_features=int(Spatial_FFN_hidden_ratio*embed_dim),
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



    def forward(self, x):
        """
        x: (N, T, H, W, C)
        local_window_pos_embed: (window_size, window_size, C)
        temporal_pos_embed: (T, C)
        Return: (N, T, H, W, C)
        """

        N, T, H, W, C = x.shape
        self.sl_posit = self.SL_POS(N=1, E=self.embed_dim, H=self.patch_size, W=self.patch_size)[0, ...].permute(1, 2, 0)
        self.sl_pos = torch.tile(self.sl_posit, (H // self.patch_size, W // self.patch_size, 1))
        self.tem_pos = self.TEM_POS(L=T, N=1, E=self.embed_dim)



        x = x + self.drop_path(self.SL_MHSA(self.norm1(x+self.sl_pos),self.norm1(x+self.sl_pos),self.norm1(x)))  # spatial local window self-attention, and skip connection

        # Conv feed-forward, different local window information interacts
        x = x + self.drop_path(self.Conv_FFN(self.norm2(x)))  # (N, T, H, W, C)

        # temporal attention
        x = x.permute(1, 0, 2, 3, 4).reshape(T, N * H * W, C)
        x = self.norm3(x)
        x = x + self.drop1(self.temporal_MHSA(self.norm3(x+self.tem_pos), self.norm3(x+self.tem_pos), self.norm3(x)))

        # output feed-forward
        x = self.norm4(x)
        x = self.linear2(self.drop2(self.activation(self.linear1(x))))
        x = x + self.drop3(x)

        x = x.reshape(T, N, H, W, C).permute(1, 0, 2, 3, 4)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    num_heads = 2
    batch_size = 32
    channel = 16
    seq_len = 15
    H = 48
    W = 48
    d_model = channel
    # K是字块大小
    Z = torch.randn(batch_size, seq_len, H, W, channel)
    Z = Z.to(device)
    encoderBlock1 = EncoderBlock(
        encH = H,
        encW = W,
        channels = channel,
        num_heads = num_heads,
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
    Z1 = encoderBlock1(Z)
    output = encoderBlock2(Z1)
    print(output.shape)