import torch
import os

# 保存多个模型部分的检查点
def save_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, checkpoint_dir, epoch):
    """
    分别保存嵌入网络、提取网络和判别器的权重及其优化器状态
    """
    # 构建保存路径，确保路径指向具体的文件而不是文件夹
    path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    os.makedirs(path, exist_ok=True)
    path_embed = os.path.join(path, "embed_net")
    path_extracted = os.path.join(path, "extract_net")
    path_discriminator = os.path.join(path, "discriminator")
    os.makedirs(path_embed, exist_ok=True)
    os.makedirs(path_extracted, exist_ok=True)
    os.makedirs(path_discriminator, exist_ok=True)
    embed_checkpoint_path = os.path.join(path_embed, f'epoch_{epoch}.pth')
    extract_checkpoint_path = os.path.join(path_extracted, f'epoch_{epoch}.pth')
    discriminator_checkpoint_path = os.path.join(path_discriminator, f'epoch_{epoch}.pth')

    save_checkpoint(embed_net, embed_optimizer, embed_checkpoint_path, epoch)
    save_checkpoint(extract_net, extract_optimizer, extract_checkpoint_path, epoch)
    save_checkpoint(discriminator, d_optimizer, discriminator_checkpoint_path, epoch)

def save_checkpoint(model, optimizer, checkpoint_path, epoch):
    """
    保存单个模型的权重及其优化器状态
    """
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except PermissionError as e:
        print(f"PermissionError: {e}. Check the file permissions for {checkpoint_path}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the checkpoint: {e}")

