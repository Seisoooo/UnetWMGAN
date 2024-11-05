import os
import torch

# 加载模型检查点
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
    # 以 epoch 编号进行排序
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)