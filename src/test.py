import os
import torch
import argparse
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SRCNN, UNet
from test_utils import test_model

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    checkpoint_path = os.path.join(config["checkpoint_dir"], 'best_checkpoint.pth')
    output_dir = config["output_dir"]

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test2017'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    srcnn = SRCNN().cuda()
    unet = UNet().cuda()

    checkpoint = torch.load(checkpoint_path)
    srcnn.load_state_dict(checkpoint['srcnn_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])

    test_model(srcnn, unet, test_loader, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
