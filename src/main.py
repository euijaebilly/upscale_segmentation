import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from srcnn import SRCNN
from unet import UNet
from data_loader import CustomDataset
import configparser
import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    miou = jaccard_score(all_labels, all_preds, average='macro')
    dice = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")


# Example usage
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.cfg',encoding='UTF-8')

    num_epochs = config.getint('TRAINING', 'num_epochs')
    batch_size = config.getint('TRAINING', 'batch_size')
    learning_rate = config.getfloat('TRAINING', 'learning_rate')
    model_type = config.get('TRAINING', 'model_type')
    hr_dir = config.get('DATA', 'train_HR_dir')
    lr_dir = config.get('DATA', 'train_LR_dir')

    train_dataset = CustomDataset(hr_dir, lr_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Select model
    if model_type == 'srcnn':
        model = SRCNN().cuda()
    elif model_type == 'unet':
        model = UNet(in_channels=3, out_channels=3).cuda()
    else:
        raise ValueError("Invalid model_type in config file")

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate model
    evaluate_model(model, train_loader)
