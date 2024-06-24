import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CocoDataset(Dataset):
    def __init__(self, origin_image_dir, data_image_dir, ann_file, transform=None, save_visualizations=False,
                 visualization_dir='check_dataset'):
        self.origin_image_dir = origin_image_dir
        self.data_image_dir = data_image_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        self.resize_transform = transforms.Resize((128, 128))
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir

        if save_visualizations:
            os.makedirs(visualization_dir, exist_ok=True)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        origin_image_path = os.path.join(self.origin_image_dir, img_info['file_name'])
        data_image_path = os.path.join(self.data_image_dir, img_info['file_name'])

        data_image = Image.open(data_image_path).convert("RGB")
        origin_image = Image.open(origin_image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        annotation_mask = self.create_mask(anns, data_image.size)
        annotation_mask = annotation_mask.convert("L")  # Convert to single channel for mask
        data_image = self.resize_transform(data_image)
        data_image = transforms.ToTensor()(data_image)
        origin_image = self.resize_transform(origin_image)
        origin_image = transforms.ToTensor()(origin_image)

        annotation_mask = self.resize_transform(annotation_mask)
        annotation_mask = transforms.ToTensor()(annotation_mask)

        # print(image.shape, annotation_mask.shape)
        # Save visualization if enabled
        if self.save_visualizations:
            self.save_visualization(data_image, annotation_mask, img_info['file_name'])

        return origin_image, data_image, annotation_mask

    def create_mask(self, anns, image_size):
        mask = Image.new('L', image_size, 0)
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list):  # polygon
                    for poly in seg:
                        poly = np.array(poly).reshape((len(poly) // 2, 2))
                        ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], outline=1, fill=1)
        return mask

    def save_visualization(self, image, annotation_mask, file_name):
        # Convert tensors to PIL images
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        if isinstance(annotation_mask, torch.Tensor):
            annotation_mask = transforms.ToPILImage()(annotation_mask)

        # Create a figure to display the images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        # Display the annotation mask
        ax[1].imshow(annotation_mask, cmap='gray')
        ax[1].set_title("Annotation Mask")
        ax[1].axis("off")

        # Save the figure
        visualization_path = os.path.join(self.visualization_dir, file_name.replace('.jpg', '_visualization.png'))
        # print(visualization_path)
        plt.savefig(visualization_path)
        plt.close(fig)
