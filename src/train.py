import torch
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader
from dataset import CocoDataset
from srcnn_unet import UNetSRCNN
from utils import train_one_epoch, validate, save_checkpoint, save_metrics, load_metrics, plot_metrics
from torchvision import transforms

# 설정 파일 로드
with open('config.json') as config_file:
    config = json.load(config_file)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor()
])

# 데이터셋 및 데이터로더 설정
train_dataset = CocoDataset(
    origin_image_dir=config['origin_train_dir'],
    data_image_dir=config['train_image_dir'],
    ann_file=config['train_ann_file'],
    transform=transform
)
val_dataset = CocoDataset(
    origin_image_dir=config['origin_val_dir'],
    data_image_dir=config['val_image_dir'],
    ann_file=config['val_ann_file'],
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# 모델 초기화
model = UNetSRCNN(1).to(device)

# 손실 함수 및 옵티마이저 설정
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion_segmentation = torch.nn.BCELoss()
criterion_sr = torch.nn.MSELoss()


# 성능 지표 파일 설정
metrics_filename = os.path.join(config['checkpoint_dir'], 'metrics.json')
metrics = load_metrics(metrics_filename)

if not metrics:
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_mious': [],
        'val_mious': [],
        'train_pixel_accuracies': [],
        'val_pixel_accuracies': [],
        'train_psrns': [],
        'val_psrns': [],
        'train_ssims': [],
        'val_ssims': []
    }

best_accuracy = 0.0

for epoch in range(len(metrics['train_losses']), config['num_epochs']):
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")

    train_loss, train_miou, train_pixel_acc, train_psrn, train_ssim = train_one_epoch(
        model, train_loader, optimizer, criterion_segmentation, criterion_sr, device)
    val_loss, val_miou, val_pixel_acc, val_psrn, val_ssim = validate(
        model, val_loader, criterion_segmentation, criterion_sr, device)

    print(f"train loss: {train_loss:.4f},mIoU: {train_miou:.4f}, Pixel Accu: {train_pixel_acc:.4f}, psrn: {train_psrn:.4f}, ssim: {train_ssim:.4f}")
    print(f"val loss: {val_loss:.4f},mIoU: {val_miou:.4f}, Pixel Accu: {val_pixel_acc:.4f}, psrn: {val_psrn:.4f}, ssim: {val_ssim:.4f}")

    metrics['train_losses'].append(train_loss)
    metrics['val_losses'].append(val_loss)
    metrics['train_mious'].append(train_miou)
    metrics['val_mious'].append(val_miou)
    metrics['train_pixel_accuracies'].append(train_pixel_acc)
    metrics['val_pixel_accuracies'].append(val_pixel_acc)
    metrics['train_psrns'].append(train_psrn)
    metrics['val_psrns'].append(val_psrn)
    metrics['train_ssims'].append(train_ssim)
    metrics['val_ssims'].append(val_ssim)

    save_metrics(metrics, metrics_filename)

    if (epoch + 1) % config['save_interval'] == 0 or epoch == config['num_epochs'] - 1:
        save_checkpoint(model, optimizer, epoch, train_loss, config['checkpoint_dir'], is_best=False)

    if train_pixel_acc > best_accuracy:
        best_accuracy = train_pixel_acc
        save_checkpoint(model, optimizer, epoch, train_loss, config['checkpoint_dir'], is_best=True)

save_checkpoint(model, optimizer, config['num_epochs'], metrics['train_losses'][-1], config['checkpoint_dir'], final=True)

# 성능 지표 그래프 저장
plot_metrics(metrics, ['losses', 'mious', 'pixel_accuracies', 'psrns', 'ssims'], config['checkpoint_dir'])


