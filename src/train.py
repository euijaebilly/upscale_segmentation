import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model import SRCNN, UNet
from utils import train, validate
from tqdm import tqdm

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

train_image_dir = config['train_image_dir']
train_ann_file = config['train_ann_file']
val_image_dir = config['val_image_dir']
val_ann_file = config['val_ann_file']
test_image_dir = config['test_image_dir']
batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['learning_rate']
patience = config['patience']
checkpoint_dir = config['checkpoint_dir']
log_dir = config['log_dir']

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets
train_dataset = CocoDataset(train_image_dir, train_ann_file, transform=transform)
val_dataset = CocoDataset(val_image_dir, val_ann_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
srcnn = SRCNN()
unet = UNet()

# Optimizers
optimizer_srcnn = optim.Adam(srcnn.parameters(), lr=learning_rate)
optimizer_unet = optim.Adam(unet.parameters(), lr=learning_rate)

# Training loop with early stopping and checkpointing
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train phase
    train_loss = train(srcnn, unet, train_loader, optimizer_srcnn, optimizer_unet)

    # Validate phase
    val_loss = validate(srcnn, unet, val_loader)

    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Checkpointing
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'srcnn_state_dict': srcnn.state_dict(),
            'unet_state_dict': unet.state_dict(),
            'optimizer_srcnn': optimizer_srcnn.state_dict(),
            'optimizer_unet': optimizer_unet.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break
