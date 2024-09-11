import numpy as np

from glob import glob
from PIL import Image
from tqdm import tqdm
from skimage.io import imread

file_directory = ""

def load_data(img_height, img_width, loaded_images):
    IMAGES_PATH = file_directory + 'images/'
    MASKS_PATH = file_directory + 'masks/'

    train_ids = sorted(glob(IMAGES_PATH + "*.jpg"))

    if loaded_images == -1:
        loaded_images = len(train_ids)

    # Initialize arrays for images and masks
    X_train = np.zeros((loaded_images, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((loaded_images, img_height, img_width), dtype=np.uint8)

    print(f"Resizing training images and masks: {loaded_images}")

    # Load and process images and masks
    for n, id_ in tqdm(enumerate(train_ids), total=loaded_images):
        if n == loaded_images:
            break

        # Load image and mask
        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = imread(image_path)
        mask = imread(mask_path)

        # Resize the image and normalize
        image_resized = np.array(Image.fromarray(image).resize((img_width, img_height)))
        X_train[n] = image_resized / 255.0

        # Convert mask to grayscale if it's in RGB
        if mask.ndim == 3:  # If the mask has 3 channels (RGB)
            mask = np.mean(mask, axis=-1)  # Convert to grayscale by averaging channels

        # Resize the mask
        mask_resized = np.array(Image.fromarray(mask).resize((img_width, img_height), resample=Image.LANCZOS))

        # Binarize the mask and store it in Y_train
        Y_train[n] = (mask_resized >= 127).astype(np.uint8)

    # Expand mask dimensions to (height, width, 1) to match the format expected by the model
    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train