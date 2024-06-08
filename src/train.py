import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CocoPreprocessedDataset
from srcnn import SRCNN
from unet import UNet


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CocoPreprocessedDataset(config["data_dir"], config["annotation_dir"], "train", transform)
    val_dataset = CocoPreprocessedDataset(config["data_dir"], config["annotation_dir"], "val", transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Initialize models
    srcnn = SRCNN().to(device)
    unet = UNet().to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(srcnn.parameters()) + list(unet.parameters()), lr=config["learning_rate"])

    # Load checkpoint if specified
    start_epoch = 0
    if config.get("checkpoint"):
        checkpoint = torch.load(config["checkpoint"])
        srcnn.load_state_dict(checkpoint['srcnn_state_dict'])
        unet.load_state_dict(checkpoint['unet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        srcnn.train()
        unet.train()
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # SRCNN forward
            outputs = srcnn(images)

            # U-Net forward
            outputs = unet(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {epoch_loss / len(train_loader)}')

        # Validation step
        srcnn.eval()
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = srcnn(images)
                outputs = unet(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(config["checkpoint_dir"], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'srcnn_state_dict': srcnn.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)
        else:
            patience_counter += 1

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config["checkpoint_dir"], f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'srcnn_state_dict': srcnn.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # Save final model
    final_model_path = os.path.join(config["checkpoint_dir"], 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'srcnn_state_dict': srcnn.state_dict(),
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)


if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    train(config)
