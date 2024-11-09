import os       
import shutil


def delete_real_png(root_dir):
    """
    删除 root_dir 下所有子文件夹中的 real.png 文件
    :param root_dir: 根目录路径
    """
    # 遍历根目录下的所有子文件夹
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 如果文件名为 real.png，则删除
            if file == 'real.png':
                file_path = os.path.join(subdir, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def distribute_images(root_dir, goal_dir):
    """
    将 goal_dir 中的 10 张图片分发到 root_dir 下的所有子文件夹中，
    每个子文件夹放入 goal_dir 中的一张图片，并将图片重命名为 real.png。
    每 10 个子文件夹为一轮，重复处理所有子文件夹。
    :param root_dir: 根目录路径
    :param goal_dir: 包含要分发图片的目录路径
    """
    # 获取 goal_dir 中的所有图片文件
    images = [f for f in os.listdir(goal_dir) if os.path.isfile(os.path.join(goal_dir, f))]
    images = images[:10]  # 仅使用前 10 张图片
    print(f"{len(images)}")
    if len(images) < 10:
        print(f"{len(images)}目标目录中的图片数量不足 10 张，请检查目标目录。")
        return

    # 遍历 root_dir 下的所有子文件夹
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 分发图片，每 10 个子文件夹为一轮
    for i, subdir in enumerate(subdirs):
        image_to_copy = images[i % 10]  # 取余数以实现每 10 个子文件夹为一轮循环
        src_image_path = os.path.join(goal_dir, image_to_copy)
        dest_image_path = os.path.join(subdir, 'real.png')
        try:
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied {src_image_path} to {dest_image_path}")
        except Exception as e:
            print(f"Error copying {src_image_path} to {dest_image_path}: {e}")

def create_folders(root_dir, num_folders=10000):
    """
    创建指定数量的文件夹，命名为 folder1-folder10000。
    :param root_dir: 根目录路径
    :param num_folders: 要创建的文件夹数量，默认为 10000
    """
    for i in range(1, num_folders + 1):
        folder_name = f"folder{i}"
        folder_path = os.path.join(root_dir, folder_name)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")

def rename_images(root_dir):
    """
    将 root_dir 的所有子文件夹中的 real.png 重命名为 watermark.png，
    将 watermark.png 重命名为 host.png。
    :param root_dir: 根目录路径
    """
    for subdir, _, files in os.walk(root_dir):
        if 'real.png' in files:
            real_path = os.path.join(subdir, 'real.png')
            watermark_path = os.path.join(subdir, 'watermark.png')
            try:
                os.rename(real_path, watermark_path)
                print(f"Renamed {real_path} to {watermark_path}")
            except Exception as e:
                print(f"Error renaming {real_path} to {watermark_path}: {e}")
        # if 'watermark.png' in files:
        #     watermark_path = os.path.join(subdir, 'watermark.png')
        #     host_path = os.path.join(subdir, 'host.png')
        #     try:
        #         os.rename(watermark_path, host_path)
        #         print(f"Renamed {watermark_path} to {host_path}")
        #     except Exception as e:
        #         print(f"Error renaming {watermark_path} to {host_path}: {e}")


# 示例用法
if __name__ == "__main__":
    
    root_directory = r"F:\UnetWMGAN_final\UnetWMGAN\dataset\tiny-imagenet-200\banana2\images"  # 将此路径替换为实际的根目录路径
    # delete_real_png(root_directory)
    # create_folders(root_directory, num_folders=10000)
    goal_directory = r"F:\UnetWMGAN_final\UnetWMGAN\dataset\tiny-imagenet-200\banana2\bananas"
    # distribute_images(root_directory, goal_directory)
    rename_images(root_directory)
