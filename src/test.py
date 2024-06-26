import torch
import json
from torchvision import transforms
from dataset import CocoDataset
from srcnn_unet import UNetSRCNN
from tqdm import tqdm
from utils import calculate_miou, calculate_pixel_accuracy, save_results

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 설정 파일 로드
with open('config.json') as config_file:
    config = json.load(config_file)

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor()
])

# 데이터셋 및 데이터로더 설정
val_dataset = CocoDataset(
    origin_image_dir=config['origin_val_dir'],
    data_image_dir=config['val_image_dir'],
    ann_file=config['val_ann_file'],
    transform=transform
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False) #, collate_fn=collate_fn)

# 모델 초기화
model = UNetSRCNN(1).to(device)

# 체크포인트 로드
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 평가 모드 설정
model.eval()

total_miou = 0
total_pixel_accuracy = 0
num_batches = 0

with torch.no_grad():
    for i, (origin_images, data_images, annotations) in enumerate(tqdm(val_loader)):
        origin_images = origin_images.to(device)
        data_images = data_images.to(device)
        annotations = annotations.to(device)

        # 모델 출력
        segmentation_outputs, sr_outputs = model(data_images)

        # 정확도 계산
        miou = calculate_miou(segmentation_outputs, annotations)
        pixel_accuracy = calculate_pixel_accuracy(segmentation_outputs, annotations)

        total_miou += miou
        total_pixel_accuracy += pixel_accuracy
        num_batches += 1

        # 결과 시각화 및 저장
        save_results(origin_images, data_images, sr_outputs, annotations, segmentation_outputs, num_batches)

average_miou = total_miou / num_batches
average_pixel_accuracy = total_pixel_accuracy / num_batches

print(f'Validation mIoU: {average_miou:.4f}')
print(f'Validation Pixel Accuracy: {average_pixel_accuracy:.4f}')
