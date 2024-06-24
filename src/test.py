import torch
from torchvision import transforms
from dataset import CocoDataset
from srcnn_unet import UNetSRCNN
from tqdm import tqdm
from utils import calculate_miou, calculate_pixel_accuracy, save_results

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
image_dir = 'preprocessed_coco/val2017'
ann_file = 'preprocessed_coco/instances_val2017.json'

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor()
])

# 데이터셋 및 데이터로더 설정
dataset = CocoDataset(image_dir=image_dir, ann_file=ann_file, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False) #, collate_fn=collate_fn)

# 모델 초기화
model = UNetSRCNN(1).to(device)

# 체크포인트 로드
checkpoint = torch.load('checkpoints/checkpoint_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 평가 모드 설정
model.eval()

total_miou = 0
total_pixel_accuracy = 0
num_batches = 0

with torch.no_grad():
    for images, annotations in tqdm(val_loader):
        images = images.to(device)
        annotations = annotations.to(device)

        # 모델 출력
        segmentation_outputs, sr_outputs = model(images)

        # 정확도 계산
        miou = calculate_miou(segmentation_outputs, annotations)
        pixel_accuracy = calculate_pixel_accuracy(segmentation_outputs, annotations)

        total_miou += miou
        total_pixel_accuracy += pixel_accuracy
        num_batches += 1

        # 결과 시각화 및 저장
        save_results(images, segmentation_outputs, sr_outputs, num_batches)

average_miou = total_miou / num_batches
average_pixel_accuracy = total_pixel_accuracy / num_batches

print(f'Validation mIoU: {average_miou:.4f}')
print(f'Validation Pixel Accuracy: {average_pixel_accuracy:.4f}')
