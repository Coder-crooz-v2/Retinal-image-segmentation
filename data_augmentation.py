import os
from glob import glob
import cv2
import numpy as np
import imageio
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from utils import create_dir

def load_data(path):
    x_train = sorted(glob(os.path.join(path, "train", "images", "*.tif")))
    y_train = sorted(glob(os.path.join(path, "train", "1st_manual", "*.gif")))
    x_test = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    y_test = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))
    return x_train, x_test, y_train, y_test

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x.split("\\")[-1].split(".")[0]
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment:
            augment = HorizontalFlip(p=1.0)
            augmented = augment(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            augment = VerticalFlip(p=1.0)
            augmented = augment(image=x1, mask=y1)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            augment = Rotate(limit=45, p=1.0)
            augmented = augment(image=x2, mask=y2)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i=cv2.resize(i, size)
            m=cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    np.random.seed(42)

    data_path = "./data"
    x_train, x_test, y_train, y_test = load_data(data_path)

    print(f"Train: x_train: {len(x_train)} - y_train: {len(y_train)}")
    print(f"Test: x_test: {len(x_test)} - y_test: {len(y_test)}")

    create_dir("augmented_data/train/image/")
    create_dir("augmented_data/train/mask/")
    create_dir("augmented_data/test/image/")
    create_dir("augmented_data/test/mask/")

    augment_data(x_train, y_train, "augmented_data/train", augment=True)
    augment_data(x_test, y_test, "augmented_data/test", augment=False)