import torch
import torch.nn as nn
from torch.autograd import Function
from .conv import Conv
import numpy as np

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        前向传播：直接返回输入
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将梯度乘以 -lambda_
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        """
        初始化梯度翻转层
        :param lambda_: 控制梯度反转的比例
        """
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """
        前向传播调用自定义的梯度翻转函数
        """
        return GradientReversalFunction.apply(x, self.lambda_)

class AdaptiveGRL(nn.Module):
    def __init__(self, initial_lambda=0.0, max_lambda=1.0, alpha=10.0, beta=0.75):
        super(AdaptiveGRL, self).__init__()
        self.register_buffer('lambda_', torch.tensor(initial_lambda))
        self.max_lambda = max_lambda
        self.alpha = alpha
        self.beta = beta
        self.iter_num = 0

    def forward(self, x):
        self.iter_num += 1
        p = float(self.iter_num) / self.alpha
        self.lambda_ = torch.tensor(self.max_lambda * (2.0 / (1.0 + np.exp(-self.beta * p)) - 1.0))
        return GradientReversalFunction.apply(x, self.lambda_)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        # 使用stride=2的卷积缩小空间尺寸，同时将通道数从in_channels变为out_channels
        # kernel_size=3, padding=1可以保持较好的特征提取特性
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downsample(x)

class DAN(nn.Module):
    def __init__(self, c1s, lambda_=0.1):
        super(DAN, self).__init__()
        self.lambda_ = lambda_
        self.grl = GRL(lambda_=lambda_)

        self.cv1 = nn.Sequential(
            GRL(self.lambda_),
            Conv(c1s[0], c1s[0], 3, 1, 1),
            DownsampleBlock(c1s[0], c1s[1]),
            Conv(c1s[1], 64, 3, 1, 1),
        )

        self.cv2 = nn.Sequential(
            GRL(self.lambda_),
            Conv(c1s[1], 256, 3, 1, 1),
            Conv(256, 64, 3, 1, 1),
        )

        self.cv3 = nn.Sequential(
            GRL(self.lambda_),
            Conv(c1s[2], 512, 3, 1, 1),
            Conv(512, 128, 3, 1, 1),
            Conv(128, 32, 3, 1, 1),
        )

        self.merge1 = nn.Sequential(
            Conv(128, 32, 3, 1, 1),
            DownsampleBlock(32, 64),
        )

        self.merge2 = nn.Sequential(
            Conv(96, 16, 3, 1, 1),
            Conv(16, 1, 3, 1, 1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 将[1,8,8] -> [1,1,1]

    def forward(self, feats):
        f1, f2, f3 = feats

        f1_out = self.cv1(f1)  
        f2_out = self.cv2(f2)  
        f3_out = self.cv3(f3)  

        merge_12 = torch.cat([f1_out, f2_out], dim=1)  
        merge_12 = self.merge1(merge_12)               

        merged_all = torch.cat([merge_12, f3_out], dim=1) 
        merged_all = self.merge2(merged_all)   
        print(f"merged_all.shape: {merged_all.shape}")            

        pooled = self.global_pool(merged_all)  
        print(f"pooled.shape: {pooled.shape}")
        domain_pred = pooled.view(pooled.size(0), -1)
        return domain_pred
