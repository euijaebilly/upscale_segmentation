import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.hr_image_files = [f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]
        self.lr_image_files = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
        assert len(self.hr_image_files) == len(self.lr_image_files), "Mismatch between HR and LR images"

    def __len__(self):
        return len(self.hr_image_files)

    def __getitem__(self, idx):
        hr_img_name = os.path.join(self.hr_dir, self.hr_image_files[idx])
        lr_img_name = os.path.join(self.lr_dir, self.lr_image_files[idx])

        hr_image = Image.open(hr_img_name).convert('RGB')
        lr_image = Image.open(lr_img_name).convert('RGB')

        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        return lr_image, hr_image
