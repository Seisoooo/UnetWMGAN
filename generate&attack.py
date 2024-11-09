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
from utils import *
from model import *

def load_config(config_path=r"F:\UnetWMGAN_final\UnetWMGAN\configs\config.yaml"):
    # 加载配置文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_psnr(image1, image2, input = "Extracted"):

    # 确保输入图像大小相同
    if image1.shape != image2.shape:
        raise ValueError("输入的两张图片大小必须相同")

    # 计算 MSE（均方误差）
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # 如果两张图片完全相同，PSNR 为无穷大

    # 计算 PSNR
    pixel_max = 255.0  # 对于 8 位图像，像素值最大为 255
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    print(f"{input}_watermark compared with original_watermrk\n PSNR = {psnr} DB")
    return psnr


def load_checkpoint(model, optimizer, checkpoint_path):
    # 加载保存的模型和优化器的状态
    try:
        checkpoint_path = os.path.abspath(checkpoint_path)  # 转为绝对路径
        checkpoint = torch.load(checkpoint_path)

        # 加载模型和优化器的状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', None)

        # 输出一些模型参数信息以验证
        if epoch is not None:
            print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
            for name, param in model.named_parameters():
                print(f"Layer: {name} | Param Sum: {param.sum().item()}")
                break  # 只打印第一个参数的值，以免输出太多
            print(f"Optimizer State Keys: {optimizer.state_dict().keys()}")
        return model, optimizer, epoch
    except PermissionError as e:
        print(f"PermissionError: {e}. Check the file permissions for {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. The checkpoint file {checkpoint_path} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None


def load_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch):
    # 分别加载嵌入网络、提取网络和判别器的权重及其优化器状态
    embed_checkpoint_path = os.path.join(checkpoint_dir, 'embed_net', f'epoch_{epoch}.pth')
    extract_checkpoint_path = os.path.join(checkpoint_dir, 'extract_net', f'epoch_{epoch}.pth')
    discriminator_checkpoint_path = os.path.join(checkpoint_dir, 'discriminator', f'epoch_{epoch}.pth')

    embed_net, embed_optimizer, loaded_epoch = load_checkpoint(embed_net, embed_optimizer, embed_checkpoint_path)
    extract_net, extract_optimizer, _ = load_checkpoint(extract_net, extract_optimizer, extract_checkpoint_path)
    discriminator, d_optimizer, _ = load_checkpoint(discriminator, d_optimizer, discriminator_checkpoint_path)

    return embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, loaded_epoch


def get_latest_checkpoint(checkpoint_dir):
    # 获取最新的检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and '_' in f]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1)
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)


def initialize():
    # 初始化模型和优化器
    config = load_config()
    embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
    extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
    discriminator = Discriminator(in_channels=3).cuda()
    embed_optimizer = torch.optim.Adam(embed_net.parameters(), lr=config['training']['embed_learning_rate'], betas=tuple(config['training']['betas']))
    extract_optimizer = torch.optim.Adam(extract_net.parameters(), lr=config['training']['extract_learning_rate'], betas=tuple(config['training']['betas']))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['training']['discriminator_learning_rate'], betas=tuple(config['training']['betas']))
    
    return embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer


def reload(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path, start_epoch=1):
    # 加载检查点
    config = load_config()
    if latest_checkpoint_path:
        try:
            latest_epoch = int(latest_checkpoint_path.split('_')[-1].split('.')[0])
            print(f"正在加载检查点：{latest_epoch}")
        except ValueError:
            print(f"Error parsing checkpoint epoch from path: {latest_checkpoint_path}")
            latest_epoch = start_epoch
        embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, start_epoch = \
            load_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path, latest_epoch)
        if embed_net is None or extract_net is None or discriminator is None:
            print("加载检查点失败，重新初始化模型")
            embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
            extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
            embed_optimizer = torch.optim.Adam(embed_net.parameters(), lr=config['training']['embed_learning_rate'], betas=tuple(config['training']['betas']))
            extract_optimizer = torch.optim.Adam(extract_net.parameters(), lr=config['training']['extract_learning_rate'], betas=tuple(config['training']['betas']))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['training']['discriminator_learning_rate'], betas=tuple(config['training']['betas']))
            start_epoch = 1      
        else:
            print(f"已成功加载检查点：{latest_epoch}") 
    else:
        print("latest_checkpoint_path为空，重新初始化模型")
    return start_epoch, embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer


def load_models():
    # 初始化并加载模型
    embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer = initialize()
    config = load_config()
    checkpoint_dir = config['paths']['checkpoint_dir']
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    start_epoch, embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer = reload(
        embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path
    )
    return embed_net, extract_net, discriminator


def load_image(image_path, transform):
    # 加载图像并应用变换
    try:
        # 检查路径是否存在
        if not os.path.exists(image_path):
            print(f"FileNotFoundError: The file {image_path} does not exist.")
            return None

        # 检查文件是否有读取权限
        if not os.access(image_path, os.R_OK):
            print(f"PermissionError: The file {image_path} cannot be read. Please check the permissions.")
            return None

        # 尝试打开图像文件
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    except PermissionError as e:
        print(f"PermissionError: {e}. Please check the permissions for {image_path}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. The file {image_path} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")

    return None



def is_watermark_image(discriminator, image, input = "extracted_watermark"):
    if isinstance(image, str):
        # 如果传入的是路径，则先加载图像
        image = Image.open(image).convert("RGB")
        transform = transforms.ToTensor()
        image = transform(image).unsqueeze(0)

    # 判别图片是否为水印图片并打印 MSE 值
    image = image.cuda()
    disc_output = discriminator(image)
    if (input != "reverse"):
        mse_value = F.mse_loss(disc_output, torch.zeros_like(disc_output))
    else : mse_value = F.mse_loss(disc_output, torch.ones_like(disc_output))

    print(f"DISTERMINATOR: {input} FAKE_LOSS: {mse_value.item()}")
    return mse_value.item()


def embed_watermark(embed_net, host_image, watermark_image, save_path):
    # 将水印图像嵌入到主图中并保存到目标文件夹
    host_image, watermark_image = host_image.cuda(), watermark_image.cuda()
    embedded_image = embed_net(host_image, watermark_image)
    embedded_image = embedded_image.cpu().detach()
    save_embed_image(embedded_image, save_path)
    print(f"Embedded image saved to {save_path}")

def extract_watermark(extract_net, embedded_image_path, transform, save_path = r"F:\UnetWMGAN_final\UnetWMGAN\test_out\results", save_name = "extracted"):
    # 从嵌入图像中提取水印并保存到目标文件夹
    embedded_image = load_image(embedded_image_path, transform)
    if embedded_image is None:
        print(f"Error: Embedded image could not be loaded from {embedded_image_path}")
        return
    
    embedded_image = embedded_image.cuda()
    extracted_watermark = extract_net(embedded_image)
    extracted_watermark = extracted_watermark.cpu().detach()
    extracted_watermark = F.interpolate(extracted_watermark, size=(64, 64))  # 调整尺寸
    save_extracted_watermark_image(extracted_watermark, save_path, save_name)
    print(f"Extracted watermark saved to {save_path}")


def save_embed_image(image, save_dir):
    fake_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)  
    fake_img = (fake_img * 255).astype('uint8')  
    img = Image.fromarray(fake_img)
    img.save(os.path.join(save_dir, f'embed.png'))

def save_extracted_watermark_image(image,save_dir = r"F:\UnetWMGAN_final\UnetWMGAN\test_out\results", save_name = "extracted"):
    real_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    real_img = (real_img * 255).astype('uint8')
    img = Image.fromarray(real_img)
    img.save(os.path.join(save_dir, f'{save_name}_watermark_image.png'))


def attack_rotate(image, angle=15):
    # 图片翻转指定角度
    return TF.rotate(image, angle, expand=True)

def attack_median_filter(image):
    # 中值滤波
    return image.filter(ImageFilter.MedianFilter(size=7))

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

def attack_blur(image, blur_radius=5, iterations=5):
    # 模糊处理，指定迭代次数来增强模糊效果
    for _ in range(iterations):
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image

def parse_and_apply_attack(embed_image_path):
    # 解析命令行参数并应用相应的攻击
    parser = argparse.ArgumentParser(description="Apply watermark attack methods to an image.")
    parser.add_argument("--image_path", type=str, default=embed_image_path, help="Path to the input image")
    parser.add_argument("--attack_type", type=str, default="random_crop", choices=["rotate", "median_filter", "random_crop", "vertical_flip", "horizontal_flip", "blur"], nargs='?', help="Type of attack to apply: choose from rotate, median_filter, random_crop, vertical_flip, horizontal_flip, blur (default: vertical_flip)")
    parser.add_argument("--angle", type=int, default=15, help="Rotation angle (for rotate attack)")
    parser.add_argument("--crop_percent", type=float, default=0.1, help="Crop percentage (for random crop attack)")
    parser.add_argument("--blur_radius", type=int, default=1, help="Blur radius (for blur attack)")
    parser.add_argument("--output_path", type=str, default="./test_out/results/attacked_embed_image.png", help="Path to save the attacked image")

    if len(sys.argv) == 1:
        print("No command line arguments provided. Defaulting to blur attack.")
    # 设置默认的图像路径
    embed_image_path = './test_out/results/embed.png'
    # 确保路径是有效的
    if not os.path.exists(embed_image_path):
        print(f"Error: The file '{embed_image_path}' does not exist.")
        sys.exit(0)
    # 解析参数
        args = parser.parse_args([embed_image_path, "blur"])
    else:
        args = parser.parse_args()


    # 加载图片
    image = Image.open(args.image_path).convert("RGB")

    # 根据攻击类型应用相应的攻击
    if args.attack_type == "rotate":
        attacked_image = attack_rotate(image, angle=args.angle)
    elif args.attack_type == "median_filter":
        attacked_image = attack_median_filter(image)
    elif args.attack_type == "random_crop":
        attacked_image = attack_random_crop(image, crop_percent=args.crop_percent)
    elif args.attack_type == "vertical_flip":
        attacked_image = attack_vertical_flip(image)
    elif args.attack_type == "horizontal_flip":
        attacked_image = attack_horizontal_flip(image)
    elif args.attack_type == "blur":
        attacked_image = attack_blur(image, blur_radius=args.blur_radius)
    else:
        raise ValueError("Invalid attack type specified.")

    # 保存攻击后的图像
    attacked_image.save(args.output_path)
    print(f"Attacked image saved to {args.output_path}")



if __name__ == "__main__":
    # 模型加载
    embed_net, extract_net, discriminator = load_models()
    
    # 示例数据路径
    host_image_path = "./test_out/host_image.png"
    watermark_image_path = "./test_out/watermark_image.png"
    embedded_save_path = r"./test_out/results"
    extracted_save_path = "./test_out/results"
    extracted_image_path = "./test_out/results/extracted_watermark_image.png"
    attacked_image_path = "./test_out/results/attacked_embed_image.png"
    attacked_extracted_image_path = r".\test_out\results\attacked_extracted_watermark_image.png"
    embedded_image_path = os.path.join(embedded_save_path, 'embed.png')

  # 确保保存目录存在
    os.makedirs(embedded_save_path, exist_ok=True)
    os.makedirs(extracted_save_path, exist_ok=True)

    # 加载图片
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    host_image = load_image(host_image_path, transform)
    watermark_image = load_image(watermark_image_path, transform)

    # 嵌入
    embed_watermark(embed_net, host_image, watermark_image, embedded_save_path)
    # 攻击
    parse_and_apply_attack(embedded_image_path)
    # 提取
    extract_watermark(extract_net, embedded_image_path, transform, extracted_save_path)
    extract_watermark(extract_net, attacked_image_path, transform, extracted_save_path, save_name= "attacked_extracted")

    is_watermark_image(discriminator, extracted_image_path)
    is_watermark_image(discriminator, attacked_image_path, input= "attacked_extracted_watermark")
    is_watermark_image(discriminator, embedded_image_path, input= "embed" )
    is_watermark_image(discriminator, host_image_path, input = "host")
    is_watermark_image(discriminator, watermark_image_path, input = "real_watermark")

    extracted_watermark_image = cv2.imread(extracted_image_path)
    attacked_extracted_image = cv2.imread(attacked_extracted_image_path)
    origin_watermark_image = cv2.imread(watermark_image_path)



    # 计算PSNR
    if extracted_watermark_image is None or origin_watermark_image is None:
        print("读取图像失败，请检查图片路径")
    else:
        psnr_value = calculate_psnr(extracted_watermark_image, origin_watermark_image)
        psnr_value = calculate_psnr(attacked_extracted_image, origin_watermark_image, input= "Attacked_Extracted")
