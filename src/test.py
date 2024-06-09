import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CocoDataset
from srcnn import SRCNN
from unet import UNet
from utils import collate_fn
from PIL import Image
import matplotlib.pyplot as plt

# Load configuration
with open('config.json') as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories and files
test_image_dir = config['test_image_dir']
checkpoint_dir = config['checkpoint_dir']

# Dataset and DataLoader
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

test_dataset = CocoDataset(image_dir=test_image_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load models
srcnn = SRCNN().to(device)
unet = UNet().to(device)

checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
checkpoint = torch.load(checkpoint_path)
srcnn.load_state_dict(checkpoint['srcnn_state_dict'])
unet.load_state_dict(checkpoint['unet_state_dict'])

srcnn.eval()
unet.eval()

# Test the models
os.makedirs('results', exist_ok=True)

with torch.no_grad():
    for idx, (images, _) in enumerate(test_loader):
        images = images.to(device)
        restored_images = srcnn(images)
        segmentation_outputs = unet(restored_images)

        # Save input, restored, and segmentation images
        input_image = transforms.ToPILImage()(images[0].cpu())
        restored_image = transforms.ToPILImage()(restored_images[0].cpu())
        segmentation_image = transforms.ToPILImage()(segmentation_outputs[0].cpu().squeeze())

        input_image.save(f'results/input_image_{idx}.png')
        restored_image.save(f'results/restored_image_{idx}.png')
        segmentation_image.save(f'results/segmentation_image_{idx}.png')

        # Plot images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(input_image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(restored_image)
        axs[1].set_title('Restored Image')
        axs[1].axis('off')

        axs[2].imshow(segmentation_image, cmap='gray')
        axs[2].set_title('Segmentation Output')
        axs[2].axis('off')

        plt.savefig(f'results/comparison_{idx}.png')
        plt.close()
