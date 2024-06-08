import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.img_annotations = {ann['image_id']: ann for ann in self.annotations['annotations']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = next(img for img in self.annotations['images'] if img['id'] == img_id)
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        annotation = self.img_annotations.get(img_id, None)

        return image, annotation
