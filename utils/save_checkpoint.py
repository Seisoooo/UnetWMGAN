import torch
import os

def save_checkpoint(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch):
    # 创建三个不同的文件夹用于分别保存三个模型
    embed_dir = os.path.join(checkpoint_dir, 'embed_net')
    extract_dir = os.path.join(checkpoint_dir, 'extract_net')
    discriminator_dir = os.path.join(checkpoint_dir, 'discriminator')
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(discriminator_dir, exist_ok=True)

    # 保存嵌入网络模型
    embed_path = os.path.join(embed_dir, f'epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': embed_net.state_dict(),
        'optimizer_state_dict': embed_optimizer.state_dict(),
    }, embed_path)
    print(f"Embed Net checkpoint saved at {embed_path}")

    # 保存提取网络模型
    extract_path = os.path.join(extract_dir, f'epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': extract_net.state_dict(),
        'optimizer_state_dict': extract_optimizer.state_dict(),
    }, extract_path)
    print(f"Extract Net checkpoint saved at {extract_path}")

    # 保存判别器模型
    discriminator_path = os.path.join(discriminator_dir, f'epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': d_optimizer.state_dict(),
    }, discriminator_path)
    print(f"Discriminator checkpoint saved at {discriminator_path}")


