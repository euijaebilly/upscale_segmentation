# Image Super-Resolution and Segmentation with PyTorch

This project contains implementations for image super-resolution using SRCNN and image segmentation using U-Net.

## Structure

- `src/main.py`: The main script for training the models.
- `src/srcnn.py`: The SRCNN model definition.
- `src/unet.py`: The U-Net model definition.
- `src/data_loader.py`: Custom dataset loader.
- `config.cfg`: Configuration file for training parameters.
- `requirements.txt`: Python dependencies.

## Usage

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Place your DIV2K training data in the specified directories in `config.cfg`.

3. Modify `config.cfg` to set your desired training parameters, including early stopping patience and checkpoint directory.

4. Run the training script:
    ```bash
    python src/main.py
    ```

5. Modify `config.cfg` to switch between SRCNN and U-Net models.

6. Check the `checkpoints` directory for saved models every 5 epochs and the `output` directory for final output images.
