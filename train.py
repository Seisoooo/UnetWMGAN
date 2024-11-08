import os
import torch
import torch.nn.functional as F
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from model import *  
from utils import *  



def train(epoch, embed_net, extract_net, discriminator, train_loader, embed_optimizer, extract_optimizer, d_optimizer, save_dir, checkpoint_dir, alpha=1.0, beta = 1.0):
    embed_net.train()
    extract_net.train()
    discriminator.train()

    for i, (watermark, host) in enumerate(train_loader):
        watermark, host = watermark.cuda(), host.cuda()
        # 判别器
        embed = embed_net(host, watermark)
        attacked = attack(embed)  # 模拟攻击
        extracted = extract_net(attacked)  
        extracted = F.interpolate(extracted, size=(watermark.size(2), watermark.size(3)))  # 调整尺寸

        disc_real = discriminator(watermark)  # 判别真实水印图像
        disc_fake = discriminator(extracted.detach())  # 判别提取水印图像

        errD_real = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real)) # 判别器对水印误差
        errD_fake = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake)) # 判别器对提取误差
        errD = errD_real + errD_fake

        d_optimizer.zero_grad()
        errD.backward()
        d_optimizer.step()

        # 嵌入网络与提取网络

        # 这里的adv_loss 与判别器的损失函数errd就相当于最基本的gan网络的两个损失函数，我们这里的判别器对应普通gan网络的判别器，要识别真实水印图片和提取的水印图片，我们这里的adv_loss这个参数就相当于普通gan网络的生成器，为了尽可能混淆判别器
        # 只不过针对这个项目我们还添加了两个必要的参数，loss_mse表示原图像与嵌入水印图像的相似性，loss_mse_extract表示能否从载密图像中提取水印图像
        # 三个参数分别对应三个目标：
        # 嵌入网络和提取网络的三个目标，1.原图像与嵌入水印图像的要相似 2.能否从载密图像中提取水印图像 3.提取的水印是能否混淆判别网络

        disc_fake = discriminator(extracted)  # 判断提取水印是否真实
        adv_loss = (1e-3) * F.binary_cross_entropy(disc_fake, torch.ones_like(disc_fake))  

        loss_mse = F.mse_loss(embed, host)  # 嵌入图像与宿主图像之间的损失
        loss_mse_extract = F.mse_loss(watermark, extracted)  # 原水印与提取水印之间的损失

        loss = loss_mse + beta * loss_mse_extract + alpha * adv_loss    
        
        embed_optimizer.zero_grad()
        extract_optimizer.zero_grad()
        loss.backward()
        embed_optimizer.step()
        extract_optimizer.step()

        # 保存图像
        if i % 4 == 0:  
            save_all_images(embed, extracted, watermark, host, attacked, epoch, i, save_dir, train_loader, loss, loss_mse, loss_mse_extract, adv_loss)

    # 保存模型
    if epoch % 5 == 0:
        save_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch)




def main():
    # 加载配置文件
    config = load_config()
    setup_logging()
    log_training_config(config)

    # 加载检查点
    checkpoint_dir = os.path.abspath(config['paths']['checkpoint_dir'])
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    embed_net, extract_net,discriminator, embed_optimizer, extract_optimizer, d_optimizer = initialize()
    start_epoch, embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer = reload(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path)

    # 加载数据集
    train_dir = config['paths']['train_dir']
    save_dir = config['paths']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('./configs/config.yaml', os.path.join(save_dir, 'config.yaml'))

    resize = tuple(config['data']['resize'])
    train_dataset = WatermarkDataset(root_dir=train_dir, transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train(epoch, embed_net, extract_net, discriminator, train_loader, embed_optimizer, extract_optimizer, d_optimizer, save_dir, checkpoint_dir, alpha=config['training']['alpha'], beta=config['training']['beta'])

if __name__ == '__main__':
    main()
