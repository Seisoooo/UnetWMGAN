import numpy as np
import torch.nn.functional as F
import torch

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