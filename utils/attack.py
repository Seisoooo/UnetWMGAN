import os
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import argparse
from PIL import Image, ImageFilter
import sys
import cv2
import numpy as np


# 频域攻击，在频域上进行微小的攻击，以后还可以尝试裁剪翻转直接贴一个logo
def attack(image):
    image_np = image.cpu().detach().numpy()
    image_fft = np.fft.fft2(image_np, axes=(-2, -1))
    # 对频域中的高频部分进行微小的扰动
    rows, cols = image_fft.shape[-2], image_fft.shape[-1]
    perturbed_fft = image_fft.copy()
    perturbed_fft[:, :, rows//4:3*rows//4, cols//4:3*cols//4] *= 1.01  # 对中心区域进行微小放大
    # 频域转换回空域
    perturbed_image_np = np.fft.ifft2(perturbed_fft, axes=(-2, -1)).real
    perturbed_image = torch.tensor(perturbed_image_np, dtype=image.dtype).cuda()
    perturbed_image = F.interpolate(perturbed_image, size=(image.size(2), image.size(3)))  
    perturbed_image = F.interpolate(perturbed_image, size=(image.size(2), image.size(3))) 
    return perturbed_image

def attack_rotate(image, angle=15):
    # 图片翻转指定角度
    return TF.rotate(image, angle, expand=True)

def attack_median_filter(image):
    # 中值滤波
    return image.filter(ImageFilter.MedianFilter(size=3))

def attack_random_crop(image, crop_percent=0.1):
    # 随机裁剪指定百分比并用空白填充
    width, height = image.size
    crop_width, crop_height = int(width * crop_percent), int(height * crop_percent)
    left = random.randint(0, crop_width)
    top = random.randint(0, crop_height)
    right = width - random.randint(0, crop_width)
    bottom = height - random.randint(0, crop_height)
    
    cropped_image = image.crop((left, top, right, bottom))
    padded_image = Image.new("RGB", (width, height), (0, 0, 0))
    padded_image.paste(cropped_image, (left, top))
    return padded_image

def attack_vertical_flip(image):
    # 垂直翻转
    return TF.vflip(image)

def attack_horizontal_flip(image):
    # 水平翻转
    return TF.hflip(image)

def attack_blur(image, blur_radius=1):
    # 模糊处理
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

if __name__ == "__main__":
    # 示例使用
    image_path = "./test_out/host_image.png"
    image = Image.open(image_path).convert("RGB")
    
    # 应用各种攻击
    rotated_15 = attack_rotate(image, 15)
    rotated_30 = attack_rotate(image, 30)
    median_filtered = attack_median_filter(image)
    random_cropped = attack_random_crop(image)
    vertical_flipped = attack_vertical_flip(image)
    horizontal_flipped = attack_horizontal_flip(image)
    blurred = attack_blur(image, blur_radius=1)
    
    # 保存攻击后的图像
    rotated_15.save("./test_out/rotated_15.png")
    rotated_30.save("./test_out/rotated_30.png")
    median_filtered.save("./test_out/median_filtered.png")
    random_cropped.save("./test_out/random_cropped.png")
    vertical_flipped.save("./test_out/vertical_flipped.png")
    horizontal_flipped.save("./test_out/horizontal_flipped.png")
    blurred.save("./test_out/blurred.png")
