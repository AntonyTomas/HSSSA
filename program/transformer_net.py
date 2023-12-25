import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding

device=torch.device('cuda:0')
# 构建Transformer预测模型
class TransformerForecasting(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerForecasting, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt):
        N,T1,C,H,W=src.size()
        N,T2,C,H,W=tgt.size()
        src = src.reshape(N, T1, C * H * W)
        tgt = tgt.reshape(N, T2, C * H * W)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        #
        src = src.permute(1, 0, 2)  # 将维度调整为（T_in，N，C*H*W）
        tgt = tgt.permute(1, 0, 2)  # 将维度调整为（T_out，N，C*H*W）
        output = self.transformer(src, tgt)  # 使用不同的序列作为输入
        output = self.fc(output)    # 维度为（N, T_out, C*H*W）
        output = output.permute(1, 0, 2)
        output = torch.reshape(output,[N,T2,C,H,W])
        return output



# 设置输入数据的维度
N, T_in, T_out, C, H, W = 16, 7, 15, 5, 48, 48
# 创建模型实例
model = TransformerForecasting(C * H * W, hidden_dim=256, num_layers=4, num_heads=8)

# 创建随机输入数据
input_data = torch.randn(N, T_in, C , H , W)
output_data = torch.randn(N, T_out, C , H , W)  # 输出序列的随机数据

# 前向传递
output = model(input_data, output_data)

print(output.shape)  # 输出预测结果的形状
