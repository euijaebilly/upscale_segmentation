import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score
import math
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

def calculate_miou(preds, labels):
    preds = (preds > 0.5).cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)
    return jaccard_score(labels.flatten(), preds.flatten(), average='binary', zero_division=1)


def calculate_pixel_accuracy(preds, labels):
    preds = (preds > 0.5).cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)
    correct = (preds == labels).sum()
    total = labels.size
    return correct / total


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1 = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.detach().permute(0, 2, 3, 1).cpu().numpy()

    ssim_values = []
    for i in range(img1.shape[0]):
        for j in range(img1.shape[-1]):
            ssim_index = ssim(img1[i, :, :, j], img2[i, :, :, j],
                              data_range=img1[i, :, :, j].max() - img1[i, :, :, j].min(),
                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            if not np.isnan(ssim_index) and not np.isinf(ssim_index):
                ssim_values.append(ssim_index)

    return np.mean(ssim_values) if ssim_values else 0


def train_one_epoch(model, data_loader, optimizer, criterion_segmentation, criterion_sr, device):
    model.train()
    running_loss = 0.0
    total_samples = len(data_loader.dataset)
    miou_total = 0.0
    pixel_acc_total = 0.0
    all_psnr = 0.0
    all_ssim = 0.0

    for i, (origin_images, data_images, annotations) in enumerate(tqdm(data_loader)):
        origin_images = origin_images.to(device)
        data_images = data_images.to(device)
        annotations = annotations.to(device)

        optimizer.zero_grad()

        segmentation_outputs, sr_outputs = model(data_images)

        loss_segmentation = criterion_segmentation(segmentation_outputs, annotations)
        loss_sr = criterion_sr(sr_outputs, origin_images)
        loss = loss_segmentation + loss_sr

        loss.backward()
        optimizer.step()

        running_loss += loss_segmentation.item() * data_images.size(0)
        miou = calculate_miou(segmentation_outputs, annotations)
        pixel_acc = calculate_pixel_accuracy(segmentation_outputs, annotations)

        miou_total += miou
        pixel_acc_total += pixel_acc
        all_psnr += calculate_psnr(sr_outputs, origin_images)
        all_ssim += calculate_ssim(sr_outputs, origin_images)
        save_image(data_images, segmentation_outputs, sr_outputs[0], 'srcnn_result', f'sr_output_train_batch_{i}.png')

    epoch_loss = running_loss / total_samples

    # 전체 평균 계산
    miou_average = miou_total / total_samples
    pixel_acc_average = pixel_acc_total / total_samples
    aver_psrn = all_psnr / total_samples
    aver_ssim = all_ssim / total_samples

    return epoch_loss, miou_average, pixel_acc_average, aver_psrn, aver_ssim


def validate(model, data_loader, criterion_segmentation, criterion_sr, device):
    model.eval()
    running_loss = 0.0
    total_samples = len(data_loader.dataset)
    miou_total = 0.0
    pixel_acc_total = 0.0
    all_psnr = 0.0
    all_ssim = 0.0

    with torch.no_grad():
        for i,(origin_images, data_images, annotations) in enumerate(tqdm(data_loader)):
            origin_images = origin_images.to(device)
            data_images = data_images.to(device)
            annotations = annotations.to(device)

            segmentation_outputs, sr_outputs = model(data_images)
            loss_segmentation = criterion_segmentation(segmentation_outputs, annotations)
            loss_sr = criterion_sr(sr_outputs, origin_images)
            loss = loss_segmentation + loss_sr

            running_loss += loss.item() * data_images.size(0)

            miou = calculate_miou(segmentation_outputs, annotations)
            pixel_acc = calculate_pixel_accuracy(segmentation_outputs, annotations)

            miou_total += miou
            pixel_acc_total += pixel_acc
            all_psnr += calculate_psnr(sr_outputs, origin_images)
            all_ssim += calculate_ssim(sr_outputs, origin_images)
            save_image(data_images, segmentation_outputs, sr_outputs[0], 'srcnn_result', f'sr_output_val_batch_{i}.png')

    epoch_loss = running_loss / total_samples

    # 전체 평균 계산
    miou_average = miou_total / total_samples
    pixel_acc_average = pixel_acc_total / total_samples
    aver_psrn = all_psnr / total_samples
    aver_ssim = all_ssim / total_samples

    return epoch_loss, miou_average, pixel_acc_average, aver_psrn, aver_ssim


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False, final=False):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    filename = 'final.pth' if final else f'checkpoint_{epoch}.pth'
    torch.save(state, os.path.join(checkpoint_dir, filename))

    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)


def load_metrics(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}


def plot_metrics(metrics, metric_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(metrics['train_losses']) + 1)

    for metric_name in metric_names:
        plt.figure()
        plt.plot(epochs, metrics[f'train_{metric_name}'], 'bo-', label=f'Training {metric_name}')
        plt.plot(epochs, metrics[f'val_{metric_name}'], 'ro-', label=f'Validation {metric_name}')
        plt.title(metric_name)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric_name}.png'))
        plt.close()


def save_results(images, segmentation_outputs, sr_outputs, batch_idx):
    os.makedirs('results', exist_ok=True)

    for i in range(images.size(0)):
        original_image = images[i].cpu().permute(1, 2, 0).numpy()
        restored_image = sr_outputs[i].cpu().permute(1, 2, 0).numpy()
        segmentation_output = segmentation_outputs[i].cpu().permute(1, 2, 0).numpy()

        overlay = np.copy(original_image)
        mask = segmentation_output[:, :, 0] > 0.5
        overlay[mask] = [1, 0, 0]

        original_image = np.clip(original_image, 0, 1)
        restored_image = np.clip(restored_image, 0, 1)
        segmentation_output = np.clip(segmentation_output, 0, 1)
        overlay = np.clip(overlay, 0, 1)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[1].imshow(restored_image)
        ax[1].set_title('Restoration Image')
        ax[2].imshow(segmentation_output)
        ax[2].set_title('Segmentation Output')
        ax[3].imshow(overlay)
        ax[3].set_title('Overlay')

        plt.savefig(f'results/result_{batch_idx}_{i}.png')
        plt.close()


def save_image(image, segmentation_outputs, sr_output, directory, filename):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    # Full path to save the image
    full_path = os.path.join(directory, filename)

    # Convert tensor to numpy array
    if image.dim() == 4:
        image = image[0]  # If there is a batch dimension, remove it
    image = image.detach().cpu().permute(1, 2, 0).numpy()  # Change to (height, width, channel)
    origin_image = np.clip(image, 0, 1)
    origin_image = (origin_image * 255.0).astype(np.uint8)

    if sr_output.dim() == 4:
        sr_output = sr_output[0]  # If there is a batch dimension, remove it
    sr_output = sr_output.detach().cpu().permute(1, 2, 0).numpy()  # Change to (height, width, channel)
    restored_image = np.clip(sr_output, 0, 1)
    restored_image = (restored_image * 255.0).astype(np.uint8)

    if segmentation_outputs.dim() == 4:
        segmentation_outputs = segmentation_outputs[0]  # If there is a batch dimension, remove it
    segmentation_outputs = segmentation_outputs.detach().cpu().permute(1, 2, 0).numpy()
    segmentation_outputs = np.clip(segmentation_outputs, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(origin_image)
    ax[0].set_title('Original Image')

    ax[1].imshow(restored_image)
    ax[1].set_title('Restoration Image')

    ax[2].imshow(segmentation_outputs)
    ax[2].set_title('segmentation_outputs')

    plt.savefig(full_path)
    plt.close()