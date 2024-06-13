import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import CocoDataset
from srcnn_unet import UNetSRCNN
from utils import collate_fn, calculate_accuracy, calculate_f1, calculate_miou, calculate_pixel_accuracy, save_results

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
image_dir = 'preprocessed_coco/val2017'
ann_file = 'preprocessed_coco/instances_val2017.json'

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 데이터셋 및 데이터로더 설정
dataset = CocoDataset(image_dir=image_dir, ann_file=ann_file, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# 모델 초기화
model = UNetSRCNN().to(device)

# 체크포인트 로드
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 평가 모드 설정
model.eval()

total_accuracy = 0
total_f1 = 0
total_miou = 0
total_pixel_accuracy = 0
num_batches = 0

with torch.no_grad():
    for images, annotations in val_loader:
        images = images.to(device)
        annotations = annotations.to(device)

        # 모델 출력
        outputs = model(images)

        # 정확도 계산
        accuracy = calculate_accuracy(outputs, annotations)
        f1 = calculate_f1(outputs, annotations)
        miou = calculate_miou(outputs, annotations)
        pixel_accuracy = calculate_pixel_accuracy(outputs, annotations)

        total_accuracy += accuracy
        total_f1 += f1
        total_miou += miou
        total_pixel_accuracy += pixel_accuracy
        num_batches += 1

        # 결과 시각화 및 저장
        save_results(images, outputs, num_batches)

average_accuracy = total_accuracy / num_batches
average_f1 = total_f1 / num_batches
average_miou = total_miou / num_batches
average_pixel_accuracy = total_pixel_accuracy / num_batches

print(f'Validation Accuracy: {average_accuracy:.4f}')
print(f'Validation F1 Score: {average_f1:.4f}')
print(f'Validation mIoU: {average_miou:.4f}')
print(f'Validation Pixel Accuracy: {average_pixel_accuracy:.4f}')
