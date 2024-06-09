import os
import random
from tqdm import tqdm
from PIL import Image, ImageFilter
import json

# Directory paths
data_dir = "dataset/coco/"
preprocessed_dir = "preprocessed_coco/"

train_dir = os.path.join(data_dir, "train2017")
val_dir = os.path.join(data_dir, "val2017")
test_dir = os.path.join(data_dir, "test2017")

train_ann_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
val_ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_images(input_dir, output_dir, annotation_file=None):
    ensure_dir(output_dir)

    if annotation_file:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

    images = [img for img in os.listdir(input_dir) if img.endswith('.jpg')]

    for img_name in tqdm(images, desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path)

        # Randomly degrade image quality
        quality = random.randint(10, 50)
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 3)))

        output_img_path = os.path.join(output_dir, img_name)
        img.save(output_img_path, "JPEG", quality=quality)

        if annotation_file:
            img_id = int(img_name.split('.')[0])
            img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
            annotations['images'].append({
                'file_name': img_name,
                'height': img.height,
                'width': img.width,
                'id': img_id
            })
            annotations['annotations'].extend(img_annotations)

    if annotation_file:
        with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f)


# Preprocess train2017
preprocess_images(train_dir, os.path.join(preprocessed_dir, "train2017"), train_ann_file)

# Preprocess val2017
preprocess_images(val_dir, os.path.join(preprocessed_dir, "val2017"), val_ann_file)

# Preprocess test2017 (without annotations)
preprocess_images(test_dir, os.path.join(preprocessed_dir, "test2017"))
