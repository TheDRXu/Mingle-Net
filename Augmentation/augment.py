import numpy as np
import albumentations as A
from albumentations.augmentations import functional as F

# Augmentation steps
aug_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    A.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
    A.GaussianBlur(blur_limit=(25, 25), sigma_limit=(0.001, 2.0), always_apply=False, p=1.0),
], additional_targets={'mask': 'mask'})

def augment_images(X_train, y_train):
    x_train_out = []
    y_train_out = []

    for i in range(len(X_train)):
        ug = aug_train(image=X_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'].astype(np.float32))  # Ensure masks are float32

    return np.array(x_train_out), np.array(y_train_out)