import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
import config


# -------------------------
# Settings
# -------------------------
SOURCE_FOLDER = f"{config.DATA_PATH}/val"   # safer than train
TARGET_FOLDER = f"{config.DATA_PATH}/external_test"
SAMPLES_PER_CLASS = 20  # you can change to 15–30


def apply_realistic_variation(image):

    # Random brightness & contrast
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Random Gaussian Blur
    if random.random() > 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Add mild Gaussian noise
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    return image


def main():

    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)

    classes = os.listdir(SOURCE_FOLDER)

    for cls in classes:

        src_class_path = os.path.join(SOURCE_FOLDER, cls)
        tgt_class_path = os.path.join(TARGET_FOLDER, cls)

        os.makedirs(tgt_class_path, exist_ok=True)

        images = os.listdir(src_class_path)
        selected = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

        print(f"Processing class: {cls}")

        for img_name in tqdm(selected):

            img_path = os.path.join(src_class_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = apply_realistic_variation(image)

            save_path = os.path.join(tgt_class_path, img_name)
            cv2.imwrite(save_path, image)

    print("\nExternal dataset created successfully!")


if __name__ == "__main__":
    main()