# Image Super-Resolution and Segmentation

This project performs image super-resolution and segmentation using SRCNN and U-Net models.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/image-super-resolution-segmentation.git
    cd image-super-resolution-segmentation
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset. Modify `config.json` to point to your COCO dataset paths.

4. Run preprocessing to degrade images:
    ```bash
    python preprocess.py
    ```

5. Train the models:
    ```bash
    python train.py
    ```

6. Test the models:
    ```bash
    python test.py
    ```

## Configuration

Modify the `config.json` file to set paths, hyperparameters, and other configurations.

## Structure

- `config.json`: Configuration file.
- `requirements.txt`: List of required packages.
- `preprocess.py`: Script to preprocess and degrade images.
- `dataset.py`: Custom dataset class for loading COCO data.
- `utils.py`: Utility functions for training and validation.
- `srcnn.py`: SRCNN model definition.
- `unet.py`: U-Net model definition.
- `train.py`: Script to train the models.
- `test.py`: Script to test the models and save results.
- `README.md`: This file.

## Results

The results, including input, restored, and segmentation images, will be saved in the `results` directory after running `test.py`.
