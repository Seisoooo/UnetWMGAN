import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from model import *  
from utils import *  

# Y配置文件
def load_config(config_path="./configs/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 模型检查点
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载保存的模型和优化器的状态
    """
    try:
        checkpoint_path = os.path.abspath(checkpoint_path)  # 转为绝对路径
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
        return model, optimizer, epoch
    except PermissionError as e:
        print(f"PermissionError: {e}. Check the file permissions for {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. The checkpoint file {checkpoint_path} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None

# 加载多个模型部分的检查点
def load_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch):
    """
    分别加载嵌入网络、提取网络和判别器的权重及其优化器状态
    """
    # 构建各自的检查点路径
    embed_checkpoint_path = os.path.join(checkpoint_dir, 'embed_net', f'epoch_{epoch}.pth')
    extract_checkpoint_path = os.path.join(checkpoint_dir, 'extract_net', f'epoch_{epoch}.pth')
    discriminator_checkpoint_path = os.path.join(checkpoint_dir, 'discriminator', f'epoch_{epoch}.pth')

    embed_net, embed_optimizer, loaded_epoch = load_checkpoint(embed_net, embed_optimizer, embed_checkpoint_path)
    extract_net, extract_optimizer, _ = load_checkpoint(extract_net, extract_optimizer, extract_checkpoint_path)
    discriminator, d_optimizer, _ = load_checkpoint(discriminator, d_optimizer, discriminator_checkpoint_path)

    return embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, loaded_epoch

# 获取最新的检查点文件
def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and '_' in f]
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)

# 训练函数
def train(epoch, embed_net, extract_net, discriminator, train_loader, embed_optimizer, extract_optimizer, d_optimizer, save_dir, checkpoint_dir, alpha=1.0):
    embed_net.train()
    extract_net.train()
    discriminator.train()

    for i, (watermark, host) in enumerate(train_loader):
        watermark, host = watermark.cuda(), host.cuda()

        embed = embed_net(host, watermark)
        attacked = attack(embed)  # 模拟攻击，得到损坏的载密图像
        extracted = extract_net(attacked)  # 提取出的水印图像
        extracted = F.interpolate(extracted, size=(watermark.size(2), watermark.size(3)))  # 调整尺寸匹配

        disc_real = discriminator(watermark)  # 判别真实水印图像
        disc_fake = discriminator(extracted.detach())  # 判别提取水印图像

        errD = adversarial_loss(disc_real, disc_fake)
        d_optimizer.zero_grad()
        errD.backward()
        d_optimizer.step()

        disc_fake = discriminator(extracted)  # 判断提取水印是否真实
        adv_loss = (1e-3) * F.binary_cross_entropy(disc_fake, torch.ones_like(disc_fake))  

        loss_mse = F.mse_loss(embed, host)  # 嵌入图像与宿主图像之间的损失
        loss_mse_extract = F.mse_loss(watermark, extracted)  # 原水印与提取水印之间的损失

        loss = loss_mse + alpha * loss_mse_extract + adv_loss
        # 对应嵌入网络和提取网络的三个目的，1.能否从载密图像中提取水印图像，2.原图像与嵌入水印图像的相似性，第三个是能否混淆判别网络
        embed_optimizer.zero_grad()
        extract_optimizer.zero_grad()
        loss.backward()
        embed_optimizer.step()
        extract_optimizer.step()

        if i % 16 == 0:  # 保存图像
            save_embed_image(embed, epoch, i, save_dir)
            save_origin_watermark_image(watermark, epoch, i, save_dir)
            save_extracted_watermark_image(extracted, epoch, i, save_dir)
            save_host_image(host, epoch, i, save_dir)
            save_attacked_image(attacked, epoch, i, save_dir)

        logging.info(f"Epoch [{epoch}/{100000}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}, MSE: {loss_mse.item()}, Extract Loss: {loss_mse_extract.item()}, Adversarial Loss: {adv_loss.item()}")

    # 保存模型
    if epoch % 50 == 0:
        save_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch)

# 保存多个模型部分的检查点
def save_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch):
    """
    分别保存嵌入网络、提取网络和判别器的权重及其优化器状态
    """
    os.makedirs(os.path.join(checkpoint_dir, 'embed_net'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'extract_net'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'discriminator'), exist_ok=True)

    embed_checkpoint_path = os.path.join(checkpoint_dir, 'embed_net', f'epoch_{epoch}.pth')
    extract_checkpoint_path = os.path.join(checkpoint_dir, 'extract_net', f'epoch_{epoch}.pth')
    discriminator_checkpoint_path = os.path.join(checkpoint_dir, 'discriminator', f'epoch_{epoch}.pth')

    save_checkpoint(embed_net, embed_optimizer, embed_checkpoint_path, epoch)
    save_checkpoint(extract_net, extract_optimizer, extract_checkpoint_path, epoch)
    save_checkpoint(discriminator, d_optimizer, discriminator_checkpoint_path, epoch)

def save_checkpoint(model, optimizer, checkpoint_path, epoch):
    """
    保存单个模型的权重及其优化器状态
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    # 配置文件
    setup_logging()
    config = load_config()
    log_training_config(config)

    # 初始化模型
    embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
    extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
    discriminator = Discriminator(in_channels=3).cuda()

    embed_optimizer = optim.Adam(embed_net.parameters(), lr=config['training']['learning_rate'], betas=tuple(config['training']['betas']))
    extract_optimizer = optim.Adam(extract_net.parameters(), lr=config['training']['learning_rate'], betas=tuple(config['training']['betas']))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config['training']['learning_rate'], betas=tuple(config['training']['betas']))

    # 加载检查点
    checkpoint_dir = os.path.abspath(config['paths']['checkpoint_dir'])
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    start_epoch = 1
    if latest_checkpoint_path:
        latest_epoch = int(latest_checkpoint_path.split('_')[-1].split('.')[0])
        print(f"加载检查点：{latest_epoch}")
        embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, start_epoch = \
            load_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, latest_epoch)
        if start_epoch is not None:
            start_epoch += 1  # 继续从下一个 epoch 开始训练

    # 加载数据集
    train_dir = config['paths']['train_dir']
    save_dir = config['paths']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    resize = tuple(config['data']['resize'])
    train_dataset = WatermarkDataset(root_dir=train_dir, transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    # 继续训练
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train(epoch, embed_net, extract_net, discriminator, train_loader, embed_optimizer, extract_optimizer, d_optimizer, save_dir, checkpoint_dir, alpha=config['training']['alpha'])

if __name__ == '__main__':
    main()
