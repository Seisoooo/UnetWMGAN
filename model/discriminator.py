import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Discriminator 模块
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 损失函数
def adversarial_loss(disc_real, disc_fake):
    errD_real = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real)) # 判别器对水印真图像接近1，提取水印图接近0
    errD_fake = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))
    errD = errD_real + errD_fake
    return errD