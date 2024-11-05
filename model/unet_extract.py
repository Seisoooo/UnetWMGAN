import torch
import torch.nn as nn
from .seblock import SEBlock

# UNetExtract
class UNetExtract(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetExtract, self).__init__()

        # 攻击后的图像预处理模块
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 16, kernel_size=4, stride=1, padding=1),  # 确保输出16通道
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2)    # 修改此层输出8通道
        )

        # SE 模块
        self.se = SEBlock(channel=8, reduction=16)

        # 缩放路径
        self.contracting_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1),  # 输入8通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 扩展路径
        self.expansive_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        preprocessed = self.preprocess(x)
        preprocessed = self.se(preprocessed)  # 加入 SE 模块
        encoded = self.contracting_path(preprocessed)  # 使用预处理后的图像进行编码
        decoded = self.expansive_path(encoded)
        return decoded


