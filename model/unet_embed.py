import torch
import torch.nn as nn
from .seblock import SEBlock
import torch.nn.functional as F

# UNetEmbed
class UNetEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEmbed, self).__init__()

        # 水印图像预处理模块
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 16, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2)
        )

        # SE 模块
        self.se = SEBlock(channel=8, reduction=16)

        # 缩放路径
        self.contracting_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels + 8, 128, kernel_size=3, stride=1, padding=1), # 3+8
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

    def forward(self, x, watermark):
        watermark_processed = self.preprocess(watermark) #
        watermark_processed = self.se(watermark_processed)  # 加入 SE 模块
        # 确保输入图像的尺寸一致
        if x.size(2) != watermark_processed.size(2) or x.size(3) != watermark_processed.size(3):
            watermark_processed = F.interpolate(watermark_processed, size=(x.size(2), x.size(3)))
        combined = torch.cat((x, watermark_processed), dim=1)  # 拼接通道，合成一张图片
        encoded = self.contracting_path(combined)
        decoded = self.expansive_path(encoded)
        return decoded
