# utils/image_utils.py

import cv2
import numpy as np
from skimage.transform import resize

def load_image(path, target_size=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img

def normalize_image(img):
    return img.astype(np.float32) / 255.0

def preprocess_image(path, target_size=(512, 512)):
    img = load_image(path, target_size)
    img = normalize_image(img)
    return img

def save_image(path, img):
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)