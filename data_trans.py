import os
import re
import torch
from torch.utils.tensorboard import SummaryWriter

# 定义日志文件路径和 TensorBoard 日志路径
log_file_path = "./training_log.txt"
tensorboard_log_dir = "./runs"

# 初始化 TensorBoard SummaryWriter
writer = SummaryWriter(tensorboard_log_dir)

# 正则表达式用于匹配日志中的各个损失项
epoch_pattern = re.compile(r'Epoch \[(\d+)/(\d+)\]')
step_pattern = re.compile(r'Step \[(\d+)/(\d+)\]')
loss_pattern = re.compile(r'Loss: ([\d\.e+-]+)')
mse_pattern = re.compile(r'MSE: ([\d\.e+-]+)')
extract_loss_pattern = re.compile(r'Extract Loss: ([\d\.e+-]+)')
adversarial_loss_pattern = re.compile(r'Adversarial Loss: ([\d\.e+-]+)')

# 读取日志文件并解析数据
with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
    for line in log_file:
        epoch_match = epoch_pattern.search(line)
        step_match = step_pattern.search(line)
        loss_match = loss_pattern.search(line)
        mse_match = mse_pattern.search(line)
        extract_loss_match = extract_loss_pattern.search(line)
        adversarial_loss_match = adversarial_loss_pattern.search(line)

        if epoch_match and step_match and loss_match and mse_match and extract_loss_match and adversarial_loss_match:
            try:
                epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                step = int(step_match.group(1))
                total_steps = int(step_match.group(2))
                loss = float(loss_match.group(1))
                mse = float(mse_match.group(1))
                extract_loss = float(extract_loss_match.group(1))
                adversarial_loss = float(adversarial_loss_match.group(1))

                # 使用全局 Step 记录进度
                global_step = epoch * total_steps + step

                # 将损失写入 TensorBoard
                writer.add_scalar('Loss/Total', loss, global_step)
                writer.add_scalar('Loss/MSE', mse, global_step)
                writer.add_scalar('Loss/Extract', extract_loss, global_step)
                writer.add_scalar('Loss/Adversarial', adversarial_loss, global_step)

            except ValueError as e:
                print(f"ValueError while parsing line: {line}")
                print(e)

# 关闭 TensorBoard SummaryWriter
writer.close()

print(f"TensorBoard logs have been saved to {tensorboard_log_dir}")