import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CocoDataset
from srcnn import SRCNN
from unet import UNet
from utils import collate_fn, train, validate
import matplotlib.pyplot as plt

# Load configuration
with open('config.json') as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories and files
train_image_dir = config['train_image_dir']
train_ann_file = config['train_ann_file']
val_image_dir = config['val_image_dir']
val_ann_file = config['val_ann_file']
checkpoint_dir = config['checkpoint_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

# Hyperparameters
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']
patience = config['patience']

# Dataset and DataLoader
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

train_dataset = CocoDataset(image_dir=train_image_dir, ann_file=train_ann_file, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_dataset = CocoDataset(image_dir=val_image_dir, ann_file=val_ann_file, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model
srcnn = SRCNN().to(device)
unet = UNet().to(device)

# Optimizers
optimizer_srcnn = torch.optim.Adam(srcnn.parameters(), lr=learning_rate)
optimizer_unet = torch.optim.Adam(unet.parameters(), lr=learning_rate)

# Loss and accuracy tracking
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training
    train_loss = train(srcnn, unet, train_loader, optimizer_srcnn, optimizer_unet, device=device)
    train_losses.append(train_loss)
    print(f"Training Loss: {train_loss:.6f}")

    # Validation
    val_loss = validate(srcnn, unet, val_loader, device=device)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.6f}")

    # Save checkpoint
    if (epoch + 1) % 5 == 0 or val_loss < best_val_loss:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'srcnn_state_dict': srcnn.state_dict(),
            'unet_state_dict': unet.state_dict(),
            'optimizer_srcnn_state_dict': optimizer_srcnn.state_dict(),
            'optimizer_unet_state_dict': optimizer_unet.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping triggered")
        break

# Save final model
final_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
torch.save({
    'srcnn_state_dict': srcnn.state_dict(),
    'unet_state_dict': unet.state_dict()
}, final_model_path)
print(f"Final model saved at {final_model_path}")

# Plot training and validation loss
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
plt.show()
