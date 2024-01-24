import torch
import torch.nn as nn
from torch import Tensor
class ConvFFN(nn.Module):

    def __init__(
        self,
        encH,
        encW,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        AR_model = True,
    ):
        super(ConvFFN,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        if AR_model:
            self.norm1 = nn.LayerNorm((hidden_features, encH, encW))
        else:
            self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        if AR_model:
            self.norm2 = nn.LayerNorm((hidden_features, encH, encW))
        else:
            self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        if AR_model:
            self.norm3 = nn.LayerNorm((out_features, encH, encW))
        else:
            self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

        self.out_features = out_features

    def forward(self, x):
        """
        x: (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x.view(N*T, H, W, C).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw3x3(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop(x)

        return x.permute(0, 2, 3, 1).reshape(N, T, H, W, self.out_features)


if __name__ == '__main__':
    num_heads = 8
    batch_size = 32
    channel = 16
    seq_len = 15
    H = 48
    W = 48

    d_model = channel
    # K是字块大小
    encH = 8
    encW = 8
    Z = torch.randn(batch_size, seq_len, H, W, channel)
    Conv_FFN_temp = ConvFFN(
        encH=48,
        encW=48,
        in_features=channel,
        hidden_features=int(4*channel),
        out_features = channel,
        drop=0.
        )

    output = Conv_FFN_temp(Z)
    print(output.shape)