import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from srcnn import SRCNN
from unet import UNet
from utils import collate_fn
from PIL import Image
import matplotlib.pyplot as plt

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

test_image_dir = config['test_image_dir']
checkpoint_dir = config['checkpoint_dir']
batch_size = config['batch_size']

# Ensure directories exist
os.makedirs(config['output_dir'], exist_ok=True)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to reduce memory usage
    transforms.ToTensor(),
])

# Load dataset
test_dataset = CocoDataset(test_image_dir, ann_file=None, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
srcnn = SRCNN().to(device)
unet = UNet().to(device)


# Find latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in the directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_checkpoint)


checkpoint_path = find_latest_checkpoint(checkpoint_dir)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
srcnn.load_state_dict(checkpoint['srcnn_state_dict'])
unet.load_state_dict(checkpoint['unet_state_dict'])

srcnn.eval()
unet.eval()


def plot_image(image, restored_image, output_path, idx):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title('Input Image')
    ax[1].imshow(restored_image.permute(1, 2, 0).cpu().numpy())
    ax[1].set_title('Restored Image')
    plt.savefig(os.path.join(output_path, f'result_{idx}.png'))
    plt.close()


# Test loop
output_dir = config['output_dir']

for idx, batch in enumerate(test_loader):
    images = batch['image'].to(device)

    with torch.no_grad():
        restored_images = srcnn(images)

    for i in range(images.size(0)):
        image = images[i]
        restored_image = restored_images[i]

        plot_image(image, restored_image, output_dir, idx * batch_size + i)

    print(f"Processed batch {idx + 1}/{len(test_loader)}")

print("Testing completed.")
