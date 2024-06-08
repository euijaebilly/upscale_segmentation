import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms

class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.ann_file = ann_file
        if ann_file:
            self.coco = COCO(ann_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.ann_file:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            image_info = self.coco.loadImgs(image_id)[0]
            path = image_info['file_name']
        else:
            path = image_id + '.jpg'

        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        annotation = None
        if self.ann_file:
            annotation = self._create_mask(anns, image.size(1), image.size(2))

        # Debug: Check shapes
        if annotation is not None:
            print(f"Image shape: {image.shape}, Annotation shape: {annotation.shape}")
        else:
            print(f"Image shape: {image.shape}, No Annotation")

        return {'image': image, 'annotation': annotation}

    def _create_mask(self, anns, target_height, target_width):
        mask = torch.zeros((target_height, target_width), dtype=torch.uint8)
        for ann in anns:
            m = self.coco.annToMask(ann)
            m = torch.tensor(m, dtype=torch.uint8)
            m = transforms.functional.resize(m.unsqueeze(0), (target_height, target_width)).squeeze(0)
            mask = torch.max(mask, m)
        return mask
