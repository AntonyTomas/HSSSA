import torch
import torch.nn as nn
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class EncoderFun(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            # 确认padding
            nn.LayerNorm([16,48,48]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0),

            nn.Conv2d(16, 64, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                # 确认padding
            nn.LayerNorm([64,48,48]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0)
        )


    def forward(self, x):
        N,T,C,H,W = x.size()
        # input = x.permute(0,1,4,2,3) # NTCHW
        input = x.reshape(N*T,C,H,W)
        output = self.conv(input)
        NT,C1,H1,W1 = output.shape
        output = output.reshape(N,T,C1,H1,W1)
        output = output.permute(0, 1, 3,4,2)#NTCHW->BTHWC
        return output


class DecoderFun(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 16 * out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            # 确认padding
            nn.LayerNorm([16 * out_channels,48,48]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0),

            nn.Conv2d(16 * out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2),
            # 确认padding
            nn.LayerNorm([out_channels,48,48]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0)
        )

    def forward(self, x):
        N, T, H, W, C = x.size()
        input = x.permute(0, 1, 4, 2, 3)  # NTHWC->NTCHW
        input = input.reshape(N * T, C, H, W)
        output = self.conv(input)

        NT, C1, H1, W1 = output.shape
        output = output.reshape(N, T, C1, H1, W1)

        return output


class EF(nn.Module):
    def __init__(self, encoderfun,encblc1,encblc2,decblc2,decblc1,decoderfun):
        super().__init__()
        self.encoderFun = encoderfun
        self.encBlock1 = encblc1
        self.encBlock2 = encblc2

        self.decBlock2 = decblc2
        self.decBlock1 = decblc1
        self.decoderFun = decoderfun

    def forward(self,x,):

        # NTCHW
        x1 = self.encoderFun(x)
        mem1 = self.encBlock1(x1)
        mem2 = self.encBlock2(mem1)
        b,t,h,w,c = mem2.size()
        query_pos = nn.Parameter(torch.randn(b,t,h,w,c))
        query_pos = query_pos.to(device)
        init_tgt = torch.zeros_like(query_pos, requires_grad=False)
        init_tgt = init_tgt.to(device)

        temp = self.decBlock2(init_tgt,query_pos,mem2)
        x2 = self.decBlock1(temp, query_pos, mem1)
        output = self.decoderFun(x2)
        return output

