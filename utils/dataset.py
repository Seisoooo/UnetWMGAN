import os
from torch.utils.data import Dataset
from PIL import Image
import random

# 数据集
class WatermarkDataset(Dataset):
    def __init__(self, root_dir, watermark_dir, transform=None):
        self.root_dir = root_dir
        self.watermark_dir = watermark_dir
        self.transform = transform
        self.all_images = self._get_all_images()
        self.watermark_images = self._get_watermark_images()

    def _get_all_images(self):
        all_images = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                images_folder = os.path.join(folder_path, 'images')
                if os.path.isdir(images_folder):
                    images = sorted(os.listdir(images_folder))
                    for img in images:
                        img_path = os.path.join(images_folder, img)
                        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            all_images.append(img_path)
        return all_images

    def _get_watermark_images(self):
        watermark_images = []
        if os.path.isdir(self.watermark_dir):
            images = sorted(os.listdir(self.watermark_dir))
            for img in images:
                img_path = os.path.join(self.watermark_dir, img)
                if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    watermark_images.append(img_path)
        return watermark_images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # 随机选择一张宿主图像
        host_img_path = self.all_images[idx]
        # 从水印文件夹中随机选择一张水印图像
        watermark_img_path = random.choice(self.watermark_images)
        try:
            host = Image.open(host_img_path).convert('RGB')
            watermark = Image.open(watermark_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            raise e

        if self.transform:
            host = self.transform(host)
            watermark = self.transform(watermark)

        return watermark, host
