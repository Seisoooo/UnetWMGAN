import os
import torch
from model import *
import yaml

# 加载配置文件
def load_config(config_path=r"F:\UnetWMGAN_final\UnetWMGAN\configs\config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 加载模型检查点
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载保存的模型和优化器的状态
    """
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
            # 输出模型的某些参数确认加载成功
            for name, param in model.named_parameters():
                print(f"Layer: {name} | Param Sum: {param.sum().item()}")
                break  # 只打印第一个参数的值，以免输出太多
            # 输出优化器的状态信息
            print(f"Optimizer State Keys: {optimizer.state_dict().keys()}")
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
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)


def initialize ():
    config = load_config()
    embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
    extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
    discriminator = Discriminator(in_channels=3).cuda()
    embed_optimizer = optim.Adam(embed_net.parameters(), lr=config['training']['embed_learning_rate'], betas=tuple(config['training']['betas']))
    extract_optimizer = optim.Adam(extract_net.parameters(), lr=config['training']['extract_learning_rate'], betas=tuple(config['training']['betas']))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config['training']['discriminator_learning_rate'], betas=tuple(config['training']['betas']))
    
    return embed_net, extract_net,discriminator, embed_optimizer, extract_optimizer, d_optimizer

def reload(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path, start_epoch = 1):
    config = load_config()
    if latest_checkpoint_path:
        latest_epoch = int(latest_checkpoint_path.split('_')[-1].split('.')[0])
        print(f"正在加载检查点：{latest_epoch}")
        embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, start_epoch = \
            load_all_checkpoints(embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer, latest_checkpoint_path, latest_epoch)
        if embed_net is None or extract_net is None or discriminator is None:
            print("加载检查点失败，重新初始化模型")
            embed_net = UNetEmbed(in_channels=3, out_channels=3).cuda()
            extract_net = UNetExtract(in_channels=3, out_channels=3).cuda()
            embed_optimizer = optim.Adam(embed_net.parameters(), lr=config['training']['embed_learning_rate'], betas=tuple(config['training']['betas']))
            extract_optimizer = optim.Adam(extract_net.parameters(), lr=config['training']['extract_learning_rate'], betas=tuple(config['training']['betas']))
            d_optimizer = optim.Adam(discriminator.parameters(), lr=config['training']['discriminator_learning_rate'], betas=tuple(config['training']['betas']))
            start_epoch = 1      
        else:
            print(f"已成功加载检查点：{latest_epoch}") 
    return start_epoch, embed_net, extract_net, discriminator, embed_optimizer, extract_optimizer, d_optimizer