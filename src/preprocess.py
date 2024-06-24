import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image

def resize_annotation(annotation, original_size, new_size):
    """
    Resize the segmentation annotations to match the new image size.
    """
    ratio_h = new_size[0] / original_size[0]
    ratio_w = new_size[1] / original_size[1]

    if 'segmentation' in annotation:
        segm = annotation['segmentation']
        if isinstance(segm, list):
            for i, seg in enumerate(segm):
                segm[i] = [coord * ratio_w if j % 2 == 0 else coord * ratio_h for j, coord in enumerate(seg)]
        elif isinstance(segm, dict):
            if isinstance(segm['counts'], list):
                segm['size'] = [int(size * ratio_h) if i == 0 else int(size * ratio_w) for i, size in
                                enumerate(segm['size'])]
        annotation['segmentation'] = segm

    if 'bbox' in annotation:
        bbox = annotation['bbox']
        bbox[0] *= ratio_w
        bbox[1] *= ratio_h
        bbox[2] *= ratio_w
        bbox[3] *= ratio_h
        annotation['bbox'] = bbox

    return annotation

def preprocess_coco(data_dir, output_dir, split):
    # COCO annotation file path
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
    coco = COCO(ann_file)

    # Create directories for preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, split)
    os.makedirs(image_output_dir, exist_ok=True)

    # Create new annotations dictionary
    new_annotations = {
        'images': [],
        'annotations': [],
        'categories': coco.loadCats(coco.getCatIds())
    }

    # Load all annotations once
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    anns_dict = {ann['image_id']: [] for ann in anns}
    for ann in anns:
        anns_dict[ann['image_id']].append(ann)

    # Iterate over all images
    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(data_dir, split, img_info['file_name'])
        img = cv2.imread(img_path)

        # Resize the image to 128x128
        resized_img = cv2.resize(img, (128, 128))
        new_img_info = img_info.copy()
        new_img_info['width'] = 128
        new_img_info['height'] = 128

        # Apply random blur or noise
        if np.random.rand() > 0.5:
            # Apply blur
            blur_factor = np.random.uniform(0.1, 1.0)
            ksize = int(blur_factor * 10) // 2 * 2 + 1
            resized_img = cv2.GaussianBlur(resized_img, (ksize, ksize), 0)
        else:
            # Apply noise
            noise_factor = np.random.uniform(0.1, 0.5)
            noise = np.random.randn(128, 128, 3) * noise_factor * 255
            resized_img = np.clip(resized_img + noise, 0, 255).astype(np.uint8)

        output_img_path = os.path.join(image_output_dir, img_info['file_name'])
        cv2.imwrite(output_img_path, resized_img)

        # Resize annotations
        if img_id in anns_dict:
            for ann in anns_dict[img_id]:
                new_ann = resize_annotation(ann, (img_info['height'], img_info['width']), (128, 128))
                new_annotations['annotations'].append(new_ann)

        new_annotations['images'].append(new_img_info)

    # Save new annotations to a JSON file
    with open(os.path.join(output_dir, f'instances_{split}.json'), 'w') as f:
        json.dump(new_annotations, f)

# Example usage
data_dir = 'dataset/coco'
output_dir = 'preprocessed_coco1'
preprocess_coco(data_dir, output_dir, split='train2017')
preprocess_coco(data_dir, output_dir, split='val2017')

