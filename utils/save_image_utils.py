import os
import torchvision.utils as vutils
from PIL import Image
import logging

def save_all_images(embed, extracted, watermark, host, attacked, epoch, i, save_dir, train_loader, loss, loss_mse, loss_mse_extract, adv_loss):
    save_embed_image(embed, epoch, i, save_dir)
    save_origin_watermark_image(watermark, epoch, i, save_dir)
    save_extracted_watermark_image(extracted, epoch, i, save_dir)
    save_host_image(host, epoch, i, save_dir)
    save_attacked_image(attacked, epoch, i, save_dir)
    logging.info(f"Epoch [{epoch}/{100000}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}, MSE: {loss_mse.item()}, Extract Loss: {loss_mse_extract.item()}, Adversarial Loss: {adv_loss.item()}")

def save_embed_image(image, epoch, step, save_dir):
    os.makedirs(os.path.join(save_dir, f'epoch_{epoch}'), exist_ok=True)
    fake_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)  
    fake_img = (fake_img * 255).astype('uint8')  
    img = Image.fromarray(fake_img)
    img.save(os.path.join(save_dir, f'epoch_{epoch}/embed.png'))

def save_host_image(image, epoch, step, save_dir):
    os.makedirs(os.path.join(save_dir, f'epoch_{epoch}'), exist_ok=True)
    real_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    real_img = (real_img * 255).astype('uint8')
    img = Image.fromarray(real_img)
    img.save(os.path.join(save_dir, f'epoch_{epoch}/origin.png'))

def save_origin_watermark_image(image, epoch, step, save_dir):
    os.makedirs(os.path.join(save_dir, f'epoch_{epoch}'), exist_ok=True)
    real_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    real_img = (real_img * 255).astype('uint8')
    img = Image.fromarray(real_img)
    img.save(os.path.join(save_dir, f'epoch_{epoch}/origin_watermark_image.png'))

def save_extracted_watermark_image(image, epoch, step, save_dir):
    os.makedirs(os.path.join(save_dir, f'epoch_{epoch}'), exist_ok=True)
    real_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    real_img = (real_img * 255).astype('uint8')
    img = Image.fromarray(real_img)
    img.save(os.path.join(save_dir, f'epoch_{epoch}/extracted_watermark_image.png'))

def save_attacked_image(image, epoch, step, save_dir):
    os.makedirs(os.path.join(save_dir, f'epoch_{epoch}'), exist_ok=True)
    real_img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    real_img = (real_img * 255).astype('uint8')
    img = Image.fromarray(real_img)
    img.save(os.path.join(save_dir, f'epoch_{epoch}/attacked_image.png'))