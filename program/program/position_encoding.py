import torch
from torch import nn
import math
import numpy as np


"""
1D position encoding and 2D postion encoding
The code is modified based on DETR of Facebook: 
https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""


class PositionEmbeddding1D(nn.Module):
    """
    1D position encoding
    Based on Attetion is all you need paper and DETR PositionEmbeddingSine class
    """

    def __init__(self, temperature=10000, normalize=False, scale=None,device=torch.device('cuda:0')):
        super().__init__()
        #设置温度参数
        self.temperature = temperature
        # scale参数scale参数用于控制位置编码中的数值范围
        # 它影响位置编码的尺度。在代码中，如果normalize参数为True，则scale参数用于对位置编码进行归一化处理。
        # normalize则代表是否进行位置编码归一化处理
        self.normalize = normalize
        self.device = device
        #如果传入了scale且normalize为False，则引发ValueError
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        #如果scale为None，则设置默认scale为2*pi
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    #前向传播函数，计算位置编码
    def forward(self, L: int, N: int, E: int):
        """
        Args:
            L for length, N for batch size, E for embedding size (dimension of transformer).

        Returns:
            pos: position encoding, with shape [L, N, E]
        """
        # 创建表示位置的矩阵，并实现累加操作
        pos_embed = torch.ones(N, L, dtype=torch.float32,device=self.device).cumsum(axis=1)
        # 创建一个维度为(E, )的向量dim_t，用于温度项计算
        dim_t = torch.arange(E, dtype=torch.float32,device=self.device)
        # 计算温度项
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / torch.div(E, 2, rounding_mode='trunc'))
        # 归一化处理位置编码
        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (L + eps) * self.scale
        # 将位置编码矩阵与温度项矩阵按元素相除，得到最终的位置编码矩阵
        pos_embed = pos_embed[:, :, None] / dim_t
        # 将位置编码矩阵与温度项矩阵按元素相除，得到最终的位置编码矩阵
        pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
        #将位置编码矩阵转置，满足Transformer输入的维度顺序要求，并标记为不可训练参数
        pos_embed = pos_embed.permute(1, 0, 2)
        pos_embed.requires_grad_(False)
        #返回最终的位置编码矩阵
        return pos_embed


class PositionEmbeddding2D(nn.Module):
    """
    2D position encoding, borrowed from DETR PositionEmbeddingSine class
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, temperature=10000, normalize=False, scale=None, device=torch.device('cuda:0')):
        super().__init__()
        # 设置温度参数
        self.temperature = temperature
        # 是否进行位置编码的归一化处理
        self.normalize = normalize
        # 设置设备，即指定使用GPU还是CPU
        self.device = device
        # 如果传入了scale且normalize为False，则引发ValueError
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 如果scale为None，则设置默认scale为2*pi
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 前向传播函数，计算2D位置编码
    def forward(self, N: int, E: int, H: int, W: int):
        """
        Args:
            N for batch size, E for embedding size (channel of feature), H for height, W for width

        Returns:
            pos_embed: positional encoding with shape (N, E, H, W)
        """
        # 判断embeding是否为偶数
        assert E % 2 == 0, "Embedding size should be even number"
        # 创建表示y方向位置的矩阵，每一行都是一个等差数列
        y_embed = torch.ones(N, H, W, dtype=torch.float32, device=self.device).cumsum(dim=1)
        # 创建表示x方向位置的矩阵，每一列都是一个等差数列
        x_embed = torch.ones(N, H, W, dtype=torch.float32, device=self.device).cumsum(dim=2)
        # 归一化处理位置编码
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        # 创建维度为(E // 2,)的向量dim_t，用于计算温度项
        dim_t = torch.arange(torch.div(E, 2, rounding_mode='trunc'), dtype=torch.float32, device=self.device)
        #计算温度项
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / torch.div(E, 2, rounding_mode='trunc'))
        # 创建维度为(E // 2,)的向量dim_t，用于计算温度项
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos_embed.requires_grad_(False)
        # 返回最终的2D位置编码矩阵
        return pos_embed

if __name__ == '__main__':
    POS =PositionEmbeddding1D()
    posit = POS(L=15, N=1, E=16)
    posit = np.array(posit)
    posit = posit[:,0,:]
    POS2D = PositionEmbeddding2D()
    posit2D = POS2D(N=1, E=16, H=48, W=48)
