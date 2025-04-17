import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms

def match_histogram_lab(dim_img, bright, save_path=None, show=False):
    # Load images in BGR
    #dim_bgr = cv2.imread(dim_img_path)
    #bright_bgr = cv2.imread(bright_img_path)

    if dim_img is None or bright is None:
        raise FileNotFoundError("One or both input image paths are invalid.")

    # Convert to LAB color space
    dim_lab = cv2.cvtColor(dim_img, cv2.COLOR_BGR2LAB)
    bright_lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)

    # Perform histogram matching on LAB
    matched_lab = match_histograms(dim_lab, bright_lab, channel_axis=-1)

    # Convert back to BGR
    matched_bgr = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


    if save_path:
        os.makedirs(os.path.dirname(save_path + '.png'), exist_ok=True)
        success = cv2.imwrite(save_path + '.png', matched_bgr)

    if show:
        dim_rgb = cv2.cvtColor(dim_img, cv2.COLOR_BGR2RGB)
        bright_rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
        matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(dim_rgb)
        axs[0].set_title("Original Dim Image")
        axs[1].imshow(bright_rgb)
        axs[1].set_title("Reference Bright Image")
        axs[2].imshow(matched_rgb)
        axs[2].set_title("LAB Histogram Matched")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return matched_bgr