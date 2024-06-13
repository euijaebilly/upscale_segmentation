import torch
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader
from dataset import CocoDataset
from srcnn_unet import UNetSRCNN
from utils import train_one_epoch, validate, collate_fn, save_checkpoint
from torchvision import transforms
import matplotlib.pyplot as plt

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
    image_dir=config['train_image_dir'],
    ann_file=config['train_ann_file'],
    transform=transform
)
val_dataset = CocoDataset(
    image_dir=config['val_image_dir'],
    ann_file=config['val_ann_file'],
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

# 모델 초기화
model = UNetSRCNN().to(device)

# 손실 함수 및 옵티마이저 설정
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion_segmentation = torch.nn.BCELoss()
criterion_sr = torch.nn.MSELoss()

# 학습 및 검증
train_losses = []
val_losses = []
accuracies = []
f1_scores = []
mious = []
pixel_accuracies = []

best_iou = 0.0

for epoch in range(config['num_epochs']):
    print(f"Epoch {epoch}/{config['num_epochs']}")

    train_loss, train_accuracy, train_f1, train_miou, train_pixel_acc = train_one_epoch(
        model, train_loader, optimizer, criterion_segmentation, criterion_sr, device)
    val_loss, val_accuracy, val_f1, val_miou, val_pixel_acc = validate(
        model, val_loader, criterion_segmentation, criterion_sr, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append((train_accuracy, val_accuracy))
    f1_scores.append((train_f1, val_f1))
    mious.append((train_miou, val_miou))
    pixel_accuracies.append((train_pixel_acc, val_pixel_acc))

    if (epoch + 1) % config['save_interval'] == 0 or epoch == config['num_epochs'] - 1:
        save_checkpoint(model, optimizer, epoch, train_loss, config['checkpoint_dir'], is_best=False)

    if val_miou > best_iou:
        best_iou = val_miou
        save_checkpoint(model, optimizer, epoch, train_loss, config['checkpoint_dir'], is_best=True)

save_checkpoint(model, optimizer, config['num_epochs'],train_losses[config['num_epochs']-1], config['checkpoint_dir'], final=True)

# 성능 지표 그래프 저장
def plot_metric(metric, metric_name, save_path):
    plt.figure()
    plt.plot(range(1, config['num_epochs'] + 1), [m[0] for m in metric], label='Train')
    plt.plot(range(1, config['num_epochs'] + 1), [m[1] for m in metric], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} over epochs')
    plt.savefig(save_path)
    plt.close()


plot_metric(train_losses, 'Loss', os.path.join(config['checkpoint_dir'], 'loss.png'))
plot_metric(accuracies, 'Accuracy', os.path.join(config['checkpoint_dir'], 'accuracy.png'))
plot_metric(f1_scores, 'F1 Score', os.path.join(config['checkpoint_dir'], 'f1_score.png'))
plot_metric(mious, 'mIoU', os.path.join(config['checkpoint_dir'], 'miou.png'))
plot_metric(pixel_accuracies, 'Pixel Accuracy', os.path.join(config['checkpoint_dir'], 'pixel_accuracy.png'))
