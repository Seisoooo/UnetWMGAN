import logging
import yaml

def load_config(config_path="./configs/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging():
    # 创建一个日志记录器
    logger = logging.getLogger()

    # 设置日志级别为 INFO
    logger.setLevel(logging.INFO)

    # 创建一个输出到控制台的Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个输出到文件的Handler
    file_handler = logging.FileHandler('training_log.txt')
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将 Handler 添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# 输出配置
def log_training_config(config):
    logging.info("训练参数:")
    logging.info(f"批次大小 (batch_size): {config['training']['batch_size']}")
    logging.info(f"训练轮数 (epochs): {config['training']['epochs']}")
    logging.info(f"学习率 (learning_rate): {config['training']['learning_rate']}")
    logging.info(f"优化器的 betas: {tuple(config['training']['betas'])}")
    logging.info(f"损失权重 (alpha): {config['training']['alpha']}")
    logging.info(f"训练数据路径 (train_dir): {config['paths']['train_dir']}")
    logging.info(f"生成图像保存路径 (save_dir): {config['paths']['save_dir']}")
    logging.info(f"检查点保存路径 (checkpoint_dir): {config['paths']['checkpoint_dir']}")
    logging.info(f"图像尺寸 (resize): {tuple(config['data']['resize'])}")