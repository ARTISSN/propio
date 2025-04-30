# utils/embedding_utils.py

import dlib
import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional
import torch

# Default paths for local execution
DEFAULT_SHAPE_PREDICTOR = "models/shape_predictor_68_face_landmarks.dat"
DEFAULT_FACE_REC_MODEL = "models/dlib_face_recognition_resnet_model_v1.dat"

# Initialize model holders
face_detector = None
shape_predictor = None
face_rec_model = None

def initialize_models(shape_predictor_path: str, face_rec_path: str):
    """Initialize dlib models lazily when needed"""
    global face_detector, shape_predictor, face_rec_model
    
    if face_detector is None:
        print("\nInitializing face detection models...")
        
        # First try environment variables
        shape_predictor_path = os.getenv("DLIB_SHAPE_PREDICTOR")
        if not face_rec_path:
            face_rec_path = os.getenv("DLIB_FACE_REC_MODEL")
        
        # If not in env vars, check local models directory
        if not shape_predictor_path or not os.path.exists(shape_predictor_path):
            # Try relative to script location
            script_dir = Path(__file__).parent.parent
            shape_predictor_path = str(script_dir / DEFAULT_SHAPE_PREDICTOR)
            face_rec_path = str(script_dir / DEFAULT_FACE_REC_MODEL)
            
            # If not there, try relative to current working directory
            if not os.path.exists(shape_predictor_path):
                shape_predictor_path = DEFAULT_SHAPE_PREDICTOR
                face_rec_path = DEFAULT_FACE_REC_MODEL
        
        # Verify models exist
        if not os.path.exists(shape_predictor_path):
            raise RuntimeError(
                f"Shape predictor model not found at {shape_predictor_path}. "
                "Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        if not os.path.exists(face_rec_path):
            raise RuntimeError(
                f"Face recognition model not found at {face_rec_path}. "
                "Please download it from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            )
            
        print(f"Using shape predictor from: {shape_predictor_path}")
        print(f"Using face recognition model from: {face_rec_path}")
        
        # Initialize models
        try:
            face_detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor(shape_predictor_path)
            face_rec_model = dlib.face_recognition_model_v1(face_rec_path)
            print("Models initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

def download_models(models_dir: str = "models"):
    """Download dlib models if they don't exist."""
    import urllib.request
    import bz2
    
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        "shape_predictor_68_face_landmarks.dat": 
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat":
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    for model_name, url in models.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            compressed_path = model_path + ".bz2"
            
            # Download compressed file
            urllib.request.urlretrieve(url, compressed_path)
            
            # Extract file
            with bz2.open(compressed_path, 'rb') as source, open(model_path, 'wb') as dest:
                dest.write(source.read())
            
            # Remove compressed file
            os.remove(compressed_path)
            print(f"Successfully downloaded and extracted {model_name}")

def get_face_embedding(
    image_input, 
    shape_predictor_path: Optional[str] = None, 
    face_rec_model_path: Optional[str] = None, 
    debug_dir: Optional[Path] = None
) -> torch.Tensor:
    """Extract 128D facial embedding from an image file path or image data."""
    # Use defaults if not provided
    if shape_predictor_path is None:
        shape_predictor_path = DEFAULT_SHAPE_PREDICTOR
    if face_rec_model_path is None:
        face_rec_model_path = DEFAULT_FACE_REC_MODEL

    try:
        initialize_models(shape_predictor_path, face_rec_model_path)  # Pass paths to initialize models
    except RuntimeError as e:
        # If models aren't found, try downloading them
        print("Models not found, attempting to download...")
        download_models()
        initialize_models(shape_predictor_path, face_rec_model_path)
    
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
    else:
        img = image_input

    # Save the image for debugging if debug_dir is provided
    if debug_dir:
        debug_image_path = debug_dir / f"debug_image_{np.random.randint(1000)}.png"
        cv2.imwrite(str(debug_image_path), img)
        print(f"Saved debug image to {debug_image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector(img_rgb, 1)

    print(f"Number of faces detected: {len(detections)}")
    if len(detections) == 0:
        print("No face found in image")
        return None

    shape = shape_predictor(img_rgb, detections[0])
    embedding = face_rec_model.compute_face_descriptor(img_rgb, shape)
    embedding_np = np.array(embedding)

    # Convert embedding to a PyTorch tensor
    embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32)

    # Optional: Save to metadata.json if image_input is a path
    if isinstance(image_input, str):
        character_path = os.path.dirname(image_input)
        if character_path:
            meta_path = os.path.join(character_path, "metadata.json")
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            base_name = os.path.splitext(os.path.basename(image_input))[0]

            # Add empty containers if not present
            if "frames" not in metadata: metadata["frames"] = {}
            if base_name not in metadata["frames"]: metadata["frames"][base_name] = {}

            metadata["frames"][base_name]["embedding"] = embedding_np.tolist()

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

    return embedding_tensor

def get_face_embedding_from_array(image_array: np.ndarray) -> np.ndarray:
    """Extract 128D facial embedding from an image array (OpenCV RGB)."""
    initialize_models()  # Ensure models are loaded
    
    detections = face_detector(image_array, 1)

    if len(detections) == 0:
        print("No face detected in image array.")
        return None

    shape = shape_predictor(image_array, detections[0])
    embedding = face_rec_model.compute_face_descriptor(image_array, shape)
    return np.array(embedding)