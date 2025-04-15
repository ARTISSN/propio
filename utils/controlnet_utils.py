# utils/controlnet_utils.py

import numpy as np
import torch

def load_normal_map(path):
    from utils.image_utils import preprocess_image
    normal = preprocess_image(path, target_size=(512, 512))
    normal_tensor = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0)
    return normal_tensor

def load_lighting_coefficients(path):
    coeffs = np.load(path)  # Assumes .npy file with shape (9,) or (27,)
    return torch.tensor(coeffs).float().unsqueeze(0)

def create_controlnet_condition(normal_map_path=None, lighting_path=None):
    normal_tensor = load_normal_map(normal_map_path) if normal_map_path else None
    lighting_tensor = load_lighting_coefficients(lighting_path) if lighting_path else None
    return {"normal_map": normal_tensor, "lighting": lighting_tensor}
