import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from tqdm import tqdm


def collate_fn(batch):
    images, annotations = zip(*batch)

    # 패딩을 위한 최종 크기
    max_height = 256
    max_width = 256

    # 패딩된 이미지와 어노테이션 생성
    padded_images = torch.zeros((len(images), 3, max_height, max_width))
    padded_annotations = torch.zeros((len(annotations), 3, max_height, max_width))

    for i, (image, annotation) in enumerate(zip(images, annotations)):
        _, h, w = image.shape
        padded_images[i, :, :h, :w] = image
        padded_annotations[i, :, :h, :w] = annotation

    return padded_images, padded_annotations


def calculate_accuracy(preds, labels):
    preds = (preds > 0.5).float()
    correct = (preds == labels).float().sum()
    total = torch.numel(preds)
    return correct / total


def calculate_f1(preds, labels):
    preds = (preds > 0.5).cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)
    return f1_score(labels.flatten(), preds.flatten(), average='binary',zero_division=1)


def calculate_miou(preds, labels):
    preds = (preds > 0.5).cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)
    return jaccard_score(labels.flatten(), preds.flatten(), average='binary',zero_division=1)


def calculate_pixel_accuracy(preds, labels):
    preds = (preds > 0.5).cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)
    correct = (preds == labels).sum()
    total = labels.size
    return correct / total


def train_one_epoch(model, data_loader, optimizer, criterion_segmentation, criterion_sr, device):
    model.train()

    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
    total_miou = 0
    total_pixel_accuracy = 0
    num_batches = 0

    for images, annotations in tqdm(data_loader):
        images = images.to(device)
        annotations = annotations.to(device)

        # 모델 출력
        outputs = model(images)
        outputs = outputs.repeat(1, 3, 1, 1)

        # print(outputs.shape,annotations.shape)

        # 손실 계산
        loss_segmentation = criterion_segmentation(outputs, annotations)
        loss_sr = criterion_sr(outputs, images)
        loss = loss_segmentation + loss_sr

        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, annotations)
        total_f1 += calculate_f1(outputs, annotations)
        total_miou += calculate_miou(outputs, annotations)
        total_pixel_accuracy += calculate_pixel_accuracy(outputs, annotations)
        num_batches += 1

    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches
    average_f1 = total_f1 / num_batches
    average_miou = total_miou / num_batches
    average_pixel_accuracy = total_pixel_accuracy / num_batches

    print(
        f'Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}, F1: {average_f1:.4f}, mIoU: {average_miou:.4f}, Pixel Accuracy: {average_pixel_accuracy:.4f}')

    return average_loss, average_accuracy, average_f1, average_miou, average_pixel_accuracy


def validate(model, data_loader, criterion_segmentation, criterion_sr, device):
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
    total_miou = 0
    total_pixel_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for images, annotations in data_loader:
            images = images.to(device)
            annotations = annotations.to(device)

            # 모델 출력
            outputs = model(images)
            outputs = outputs.repeat(1, 3, 1, 1)

            # 손실 계산
            loss_segmentation = criterion_segmentation(outputs, annotations)
            loss_sr = criterion_sr(outputs, images)
            loss = loss_segmentation + loss_sr

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, annotations)
            total_f1 += calculate_f1(outputs, annotations)
            total_miou += calculate_miou(outputs, annotations)
            total_pixel_accuracy += calculate_pixel_accuracy(outputs, annotations)
            num_batches += 1

    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches
    average_f1 = total_f1 / num_batches
    average_miou = total_miou / num_batches
    average_pixel_accuracy = total_pixel_accuracy / num_batches

    print(
        f'Validation Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}, F1: {average_f1:.4f}, mIoU: {average_miou:.4f}, Pixel Accuracy: {average_pixel_accuracy:.4f}')

    return average_loss, average_accuracy, average_f1, average_miou, average_pixel_accuracy


def save_results(images, outputs, batch_idx):
    os.makedirs('results', exist_ok=True)

    for i in range(images.size(0)):
        original_image = images[i].cpu().permute(1, 2, 0).numpy()
        segmentation_output = outputs[i].cpu().permute(1, 2, 0).numpy()

        # 오버레이 결과 생성
        overlay = np.copy(original_image)
        mask = segmentation_output[:, :, 0] > 0.5  # 첫 번째 채널을 기준으로 마스크 생성
        overlay[mask] = [1, 0, 0]  # 예시로 빨간색을 세그멘테이션 결과로 표시

        # 범위 조정
        original_image = np.clip(original_image, 0, 1)
        segmentation_output = np.clip(segmentation_output, 0, 1)
        overlay = np.clip(overlay, 0, 1)

        # 시각화
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[1].imshow(segmentation_output)
        ax[1].set_title('Segmentation Output')
        ax[2].imshow(overlay)
        ax[2].set_title('Overlay')

        # 파일 저장
        plt.savefig(f'results/result_{batch_idx}_{i}.png')
        plt.close()


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


def plot_metrics(train_losses, val_losses, accuracies, f1_scores, mious, pixel_accuracies):
    os.makedirs('checkpoints', exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, accuracies, 'bo-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, f1_scores, 'bo-', label='Validation F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, mious, 'bo-', label='Validation mIoU')
    plt.plot(epochs, pixel_accuracies, 'ro-', label='Validation Pixel Accuracy')
    plt.title('mIoU and Pixel Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('checkpoints', 'metrics.png'))
    plt.close()
