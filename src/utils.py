import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def collate_fn(batch):
    images = [item['image'] for item in batch]
    annotations = [item['annotation'] for item in batch if 'annotation' in item]

    # Find the maximum width and height in the batch
    max_width = max(image.shape[2] for image in images)
    max_height = max(image.shape[1] for image in images)

    # Pad all images to the maximum width and height
    padded_images = []
    for image in images:
        padding = (0, max_width - image.shape[2], 0, max_height - image.shape[1])
        padded_image = torch.nn.functional.pad(image, padding, value=0)
        padded_images.append(padded_image)

    images = torch.stack(padded_images)

    if annotations:
        max_anno_width = max(anno.shape[1] for anno in annotations)
        max_anno_height = max(anno.shape[0] for anno in annotations)

        # Pad all annotations to the maximum width and height
        padded_annotations = []
        for anno in annotations:
            padding = (0, max_anno_width - anno.shape[1], 0, max_anno_height - anno.shape[0])
            padded_annotation = torch.nn.functional.pad(anno, padding, value=0)
            padded_annotations.append(padded_annotation)

        annotations = torch.stack(padded_annotations)
    else:
        annotations = None

    return images, annotations



def train(srcnn, unet, data_loader, optimizer_srcnn, optimizer_unet, device='cuda'):
    srcnn.train()
    unet.train()
    total_loss = 0
    for images, annotations in tqdm(data_loader, desc="Training"):
        images = images.to(device)

        optimizer_srcnn.zero_grad()
        restored_images = srcnn(images)
        restored_images = restored_images.to(device)

        optimizer_unet.zero_grad()
        outputs = unet(restored_images)

        loss = F.mse_loss(outputs, annotations)
        loss.backward()
        optimizer_srcnn.step()
        optimizer_unet.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(srcnn, unet, data_loader, device='cuda'):
    srcnn.eval()
    unet.eval()
    total_loss = 0
    with torch.no_grad():
        for images, annotations in tqdm(data_loader, desc="Validating"):
            images = images.to(device)
            restored_images = srcnn(images)
            restored_images = restored_images.to(device)
            outputs = unet(restored_images)

            loss = F.mse_loss(outputs, annotations)

            total_loss += loss.item()

    return total_loss / len(data_loader)