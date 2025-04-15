# utils/embedding_utils.py

import dlib
import cv2
import numpy as np
import os

# Paths to Dlib models (ensure these are downloaded and correctly placed)
SHAPE_PREDICTOR_PATH = os.getenv("DLIB_SHAPE_PREDICTOR", "models/shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.getenv("DLIB_FACE_REC_MODEL", "models/dlib_face_recognition_resnet_model_v1.dat")

# Load Dlib models globally (once)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)

def get_face_embedding(image_path: str) -> np.ndarray:
    """Extract 128D facial embedding from an image file path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector(img_rgb, 1)

    if len(detections) == 0:
        print(f"No face found in {image_path}")
        return None

    shape = shape_predictor(img_rgb, detections[0])
    embedding = face_rec_model.compute_face_descriptor(img_rgb, shape)
    return np.array(embedding)

def get_face_embedding_from_array(image_array: np.ndarray) -> np.ndarray:
    """Extract 128D facial embedding from an image array (OpenCV RGB)."""
    detections = face_detector(image_array, 1)

    if len(detections) == 0:
        print("No face detected in image array.")
        return None

    shape = shape_predictor(image_array, detections[0])
    embedding = face_rec_model.compute_face_descriptor(image_array, shape)
    return np.array(embedding)