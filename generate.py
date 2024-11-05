import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from model import UNetEmbed, UNetExtract, Discriminator  # 假设你的模型定义在 model.py 中
import torch.nn.functional as F

# 加载 YAML 配置文件
def load_config(config_path="./configs/config.yaml"):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 加载模型的检查点
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# 图像加载辅助函数
def load_image(image_path, resize=None):
    image = Image.open(image_path).convert('RGB')
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)  # Add batch dimension

# 保存图片辅助函数
def save_image(tensor, save_path):
    image = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image = (image * 255).astype('uint8')
    Image.fromarray(image).save(save_path)

def embed_watermark(embed_model, host_image_path, watermark_image_path, output_path, config):
    host_image = load_image(host_image_path, resize=config['data']['resize']).cuda()
    watermark_image = load_image(watermark_image_path, resize=config['data']['resize']).cuda()
    
    # 嵌入水印
    embedded_image = embed_model(host_image, watermark_image)
    
    # 保存生成的图片
    save_image(embedded_image, output_path)
    print(f"水印嵌入完成，保存到 {output_path}")

def extract_watermark(extract_model, input_image_path, output_path, config):
    input_image = load_image(input_image_path, resize=config['data']['resize']).cuda()
    
    # 提取水印
    extracted_watermark = extract_model(input_image)
    extracted_watermark = F.interpolate(extracted_watermark, size=config['data']['resize'])  # 调整大小
    
    # 保存提取的水印图片
    save_image(extracted_watermark, output_path)
    print(f"水印提取完成，保存到 {output_path}")

def discriminate_watermark(discriminator_model, input_image_path, config):
    input_image = load_image(input_image_path, resize=config['data']['resize']).cuda()
    
    # 使用判别器判断是否有水印
    result = discriminator_model(input_image)
    result = torch.sigmoid(result).item()  # 判别结果是一个概率值

    if result > 0.5:
        print(f"图片中检测到水印，置信度：{result:.2f}")
    else:
        print(f"图片中未检测到水印，置信度：{result:.2f}")

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="水印生成、提取和辨别工具")
    parser.add_argument("operation", choices=["emb", "extract", "discriminate"], help="操作类型: emb (嵌入水印), extract (提取水印), discriminate (辨别水印)")
    parser.add_argument("input1", help="输入图片路径")
    parser.add_argument("input2", nargs="?", default=None, help="第二个输入图片路径，仅在嵌入水印时需要 (水印图片)")
    parser.add_argument("--output", default="./output.png", help="输出图片路径 (默认: ./output.png)")
    parser.add_argument("--config", default="./configs/config.yaml", help="配置文件路径 (默认: ./configs/config.yaml)")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/epoch_650.pth", help="检查点目录路径 (默认: ./checkpoints)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # 根据操作类型选择对应的模型
    if args.operation == "emb":
        if not args.input2:
            raise ValueError("嵌入水印操作需要两个输入图片：宿主图片和水印图片")

        embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
        embed_checkpoint_path = os.path.join(checkpoint_dir, 'embed_net')  # 修改为你需要的检查点
        embed_net = load_model(embed_net, embed_checkpoint_path)
        
        embed_watermark(embed_net, args.input1, args.input2, args.output, config)

    elif args.operation == "extract":
        extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
        extract_checkpoint_path = os.path.join(checkpoint_dir, 'extract_net')  # 修改为你需要的检查点
        extract_net = load_model(extract_net, extract_checkpoint_path)
        
        extract_watermark(extract_net, args.input1, args.output, config)

    elif args.operation == "discriminate":
        discriminator = Discriminator(in_channels=3).cuda()
        discriminator_checkpoint_path = os.path.join(checkpoint_dir, 'discriminator')  # 修改为你需要的检查点
        discriminator = load_model(discriminator, discriminator_checkpoint_path)
        
        discriminate_watermark(discriminator, args.input1, config)

if __name__ == "__main__":
    main()
