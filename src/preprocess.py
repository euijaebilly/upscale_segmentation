import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def preprocess_coco(data_dir, output_dir, split='train2017'):
    # COCO annotation file path
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')

    # Initialize COCO API
    coco = COCO(ann_file)

    # Create directories for preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, split)
    os.makedirs(image_output_dir, exist_ok=True)
    check_dir = os.path.join(output_dir, 'check_dataset')
    os.makedirs(check_dir, exist_ok=True)

    # Iterate over all images
    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(data_dir, split, img_info['file_name'])
        img = cv2.imread(img_path)

        # Apply random blur
        blur_factor = np.random.uniform(0.1, 0.9)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        output_img_path = os.path.join(image_output_dir, img_info['file_name'])
        cv2.imwrite(output_img_path, img)

        # Create segmentation mask
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    for poly in seg:
                        poly = np.array(poly).reshape((len(poly) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Save the mask
        # mask_output_path = os.path.join(image_output_dir, f'{img_info["file_name"].split(".")[0]}_mask.png')
        # cv2.imwrite(mask_output_path, mask * 255)

        # Visualize and save images
        # original_image = Image.open(img_path)
        # blurred_image = Image.open(output_img_path)
        # mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        # check_image = Image.new('RGB', (original_image.width * 3, original_image.height))
        # check_image.paste(original_image, (0, 0))
        # check_image.paste(blurred_image, (original_image.width, 0))
        # check_image.paste(mask_image, (original_image.width * 2, 0))
        # check_image.save(os.path.join(check_dir, f'{img_info["file_name"].split(".")[0]}_check.png'))


# Example usage
data_dir = 'dataset/coco'
output_dir = 'preprocessed_coco'
preprocess_coco(data_dir, output_dir, split='train2017')
preprocess_coco(data_dir, output_dir, split='val2017')
