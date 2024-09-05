import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Correlation(torch.nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, x):
        # 获取张量的维度
        N, C, H, W = x.size()

        x_reshaped = x.view(N, C, -1)  # 形状: (N, C, H*W)
        x_mean = x_reshaped.mean(dim=2, keepdim=True)  # 平均值 (N, C, 1)
        x_centered = x_reshaped - x_mean  # 中心化 (N, C, H*W)
        x_norm = torch.norm(x_centered, dim=2, keepdim=True)  # 归一化因子 (N, C, 1)
        x_normalized = x_centered / (x_norm + 1e-8)  # 归一化 (N, C, H*W)

        # 计算皮尔逊相关系数
        correlation_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # (N, C, C)

        # 对角线上的值即为每个通道与自身的相关系数，将其置为0
        for i in range(C):
            correlation_matrix[:, i, i] = 0

        # 取绝对值并计算总和，然后除以通道数的组合数以得到平均值
        absolute_correlation = torch.abs(correlation_matrix)
        total_correlation = absolute_correlation.sum()
        num_combinations = len(list(itertools.combinations(range(C), 2)))  # 通道数的组合数
        average_correlation = total_correlation / num_combinations / 2 / N

        return average_correlation