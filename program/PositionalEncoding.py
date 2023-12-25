import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
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
        # pos_embed = pos_embed.permute(1, 0, 2)
        pos_embed.requires_grad_(False)
        #返回最终的位置编码矩阵
        return pos_embed
