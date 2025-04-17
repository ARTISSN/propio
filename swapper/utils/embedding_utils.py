# utils/embedding_utils.py

import dlib
import cv2
import numpy as np
import os
import json

# Paths to Dlib models
SHAPE_PREDICTOR_PATH = os.getenv("DLIB_SHAPE_PREDICTOR", "models/shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.getenv("DLIB_FACE_REC_MODEL", "models/dlib_face_recognition_resnet_model_v1.dat")

# Initialize model holders
face_detector = None
shape_predictor = None
face_rec_model = None

def initialize_models():
    """Initialize dlib models lazily when needed"""
    global face_detector, shape_predictor, face_rec_model
    
    if face_detector is None:
        print("\nInitializing face detection models...")
        print(f"Looking for shape predictor at: {SHAPE_PREDICTOR_PATH}")
        print(f"Path exists: {os.path.exists(SHAPE_PREDICTOR_PATH)}")
        
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            raise RuntimeError(f"Shape predictor model not found at {SHAPE_PREDICTOR_PATH}")
            
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
        print("Models initialized successfully")

def get_face_embedding(image_path: str, character_path: str = None) -> np.ndarray:
    """Extract 128D facial embedding from an image file path."""
    initialize_models()  # Ensure models are loaded
    
    print(f"\nProcessing image: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
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
    embedding_np = np.array(embedding)

    # Optional: Save to metadata.json
    if character_path is not None:
        meta_path = os.path.join(character_path, "metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Add empty containers if not present
        if "frames" not in metadata: metadata["frames"] = {}
        if base_name not in metadata["frames"]: metadata["frames"][base_name] = {}

        metadata["frames"][base_name]["embedding"] = embedding_np.tolist()

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return embedding_np

def get_face_embedding_from_array(image_array: np.ndarray) -> np.ndarray:
    """Extract 128D facial embedding from an image array (OpenCV RGB)."""
    detections = face_detector(image_array, 1)

    if len(detections) == 0:
        print("No face detected in image array.")
        return None

    shape = shape_predictor(image_array, detections[0])
    embedding = face_rec_model.compute_face_descriptor(image_array, shape)
    return np.array(embedding)