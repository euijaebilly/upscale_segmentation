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
    img1 = img1 / 255.0 if img1.max() > 1 else img1
    img2 = img2 / 255.0 if img2.max() > 1 else img2
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1 = img1 / 255.0 if img1.max() > 1 else img1
    img2 = img2 / 255.0 if img2.max() > 1 else img2
    img1 = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.detach().permute(0, 2, 3, 1).cpu().numpy()

    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_index = ssim(img1[i], img2[i], data_range=img1[i].max() - img1[i].min(),
                          win_size=7, channel_axis=-1)
        if not np.isnan(ssim_index) and not np.isinf(ssim_index):
            ssim_values.append(ssim_index)

    return np.mean(ssim_values) if ssim_values else 0


def train_one_epoch(model, data_loader, optimizer, criterion_segmentation, criterion_sr, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
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

        batch_size = data_images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        miou_total += calculate_miou(segmentation_outputs, annotations) * batch_size
        pixel_acc_total += calculate_pixel_accuracy(segmentation_outputs, annotations) * batch_size
        all_psnr += calculate_psnr(sr_outputs, origin_images) * batch_size
        all_ssim += calculate_ssim(sr_outputs, origin_images) * batch_size
        save_image(data_images, segmentation_outputs, sr_outputs, 'srcnn_result', f'sr_output_train_batch_{i}.png')

    epoch_loss = running_loss / total_samples
    miou_average = miou_total / total_samples
    pixel_acc_average = pixel_acc_total / total_samples
    average_psnr = all_psnr / total_samples
    average_ssim = all_ssim / total_samples

    return epoch_loss, miou_average, pixel_acc_average, average_psnr, average_ssim


def validate(model, data_loader, criterion_segmentation, criterion_sr, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    miou_total = 0.0
    pixel_acc_total = 0.0
    all_psnr = 0.0
    all_ssim = 0.0

    with torch.no_grad():
        for i, (origin_images, data_images, annotations) in enumerate(tqdm(data_loader)):
            origin_images = origin_images.to(device)
            data_images = data_images.to(device)
            annotations = annotations.to(device)

            segmentation_outputs, sr_outputs = model(data_images)
            loss_segmentation = criterion_segmentation(segmentation_outputs, annotations)
            loss_sr = criterion_sr(sr_outputs, origin_images)
            loss = loss_segmentation + loss_sr

            batch_size = data_images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            miou_total += calculate_miou(segmentation_outputs, annotations) * batch_size
            pixel_acc_total += calculate_pixel_accuracy(segmentation_outputs, annotations) * batch_size
            all_psnr += calculate_psnr(sr_outputs, origin_images) * batch_size
            all_ssim += calculate_ssim(sr_outputs, origin_images) * batch_size
            save_image(data_images, segmentation_outputs, sr_outputs, 'srcnn_result', f'sr_output_val_batch_{i}.png')

    epoch_loss = running_loss / total_samples
    miou_average = miou_total / total_samples
    pixel_acc_average = pixel_acc_total / total_samples
    average_psnr = all_psnr / total_samples
    average_ssim = all_ssim / total_samples

    return epoch_loss, miou_average, pixel_acc_average, average_psnr, average_ssim


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


def save_results(origin_images, data_images, sr_outputs, annotations, segmentation_outputs, batch_idx):
    os.makedirs('results', exist_ok=True)

    for i in range(data_images.size(0)):
        # Convert tensors to numpy arrays
        origin_image = origin_images[i].cpu().permute(1, 2, 0).numpy()
        data_image = data_images[i].cpu().permute(1, 2, 0).numpy()
        sr_image = sr_outputs[i].cpu().permute(1, 2, 0).numpy()
        annotation = annotations[i].cpu().permute(1, 2, 0).numpy()
        segmentation_output = segmentation_outputs[i].cpu().permute(1, 2, 0).numpy()

        # Ensure values are in range [0, 1]
        origin_image = np.clip(origin_image, 0, 1)
        data_image = np.clip(data_image, 0, 1)
        sr_image = np.clip(sr_image, 0, 1)
        annotation = np.clip(annotation, 0, 1)
        segmentation_output = np.clip(segmentation_output, 0, 1)
        print(f"annotation Output Example Values: {annotation[0, 0, 0]}, {annotation[0, 1, 0]}, {annotation[1, 0, 0]}")
        print(f"Segmentation Output Example Values: {segmentation_output[0, 0, 0]}, {segmentation_output[0, 1, 0]}, {segmentation_output[1, 0, 0]}")

        # Create masks for overlays
        annotation_mask = annotation[:, :, 0] > 0.001
        segmentation_mask = segmentation_output[:, :, 0] > 0.001

        # Overlay masks on data images
        overlay_annotation = data_image.copy()
        overlay_annotation[annotation_mask] = [1, 0, 0]  # Red for original annotation

        overlay_segmentation = data_image.copy()
        overlay_segmentation[segmentation_mask] = [0, 1, 0]  # Green for predicted segmentation

        # Plot all images
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax[0,0].imshow(origin_image)
        ax[0,0].set_title('Original Image')

        ax[0,1].imshow(data_image)
        ax[0,1].set_title('Data Image')

        ax[0,2].imshow(sr_image)
        ax[0,2].set_title('Restored Image')

        ax[1,0].imshow(annotation[:, :, 0], cmap='gray')
        ax[1,0].set_title('Original Annotation')

        ax[1,1].imshow(segmentation_output[:, :, 0], cmap='gray')
        ax[1,1].set_title('Segmentation Output')

        ax[1,2].imshow(overlay_annotation)
        ax[1,2].set_title('Overlay Annotation')

        ax[1,3].imshow(overlay_segmentation)
        ax[1,3].set_title('Overlay Segmentation')

        # Remove the last empty subplot
        fig.delaxes(ax[0, 3])

        for a in ax.flat:
            a.axis('off')

        plt.tight_layout()
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
