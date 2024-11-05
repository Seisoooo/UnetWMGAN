
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# 数据集
class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._get_image_pairs()

    def _get_image_pairs(self):
        image_pairs = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                xh_path = os.path.join(folder_path, 'real.png')
                xs_path = os.path.join(folder_path, 'watermark.png')

                if os.path.exists(xh_path) and os.path.exists(xs_path):
                    image_pairs.append((xh_path, xs_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        xh_path, xs_path = self.image_pairs[idx]
        try:
            host = Image.open(xh_path).convert('RGB')
            watermark = Image.open(xs_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            raise e

        if self.transform:
            host = self.transform(host)
            watermark = self.transform(watermark)

        return watermark, host