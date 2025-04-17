import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from scipy.special import sph_harm
from pathlib import Path
import datetime

def compute_spherical_harmonics_basis(normals, order=2):
    """Compute spherical harmonics basis functions up to specified order."""
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]

    # Compute spherical coordinates
    theta = np.arccos(np.clip(z, -1.0, 1.0))  # Elevation angle from Z-axis
    phi = np.arctan2(y, x)                     # Azimuth angle in XY-plane

    # Initialize basis functions array
    num_coeffs = (order + 1) ** 2
    basis = np.zeros((len(normals), num_coeffs))

    # Compute SH basis functions
    idx = 0
    for l in range(order + 1):
        for m in range(-l, l + 1):
            basis[:, idx] = np.real(sph_harm(m, l, phi, theta))
            idx += 1

    return basis

def calculate_lighting_coefficients(face_img, normal_map, order=3):
    """Calculate lighting coefficients for a face image and its normal map."""
    # Convert normal map from BGR to RGB and normalize to [-1, 1]
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    normal_map = 2.0 * (normal_map.astype(np.float32) / 255.0) - 1.0
    
    # Convert face image to grayscale and normalize to [0, 1]
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype(np.float32) / 255.0
    
    # Create mask for valid pixels
    mask = np.sum(np.abs(normal_map), axis=2) > 0.1
    
    # Extract valid pixels
    valid_pixels = face_img[mask]
    valid_normals = normal_map[mask].reshape(-1, 3)
    
    # Sample pixels if there are too many
    if len(valid_pixels) > 10000:
        indices = np.random.choice(len(valid_pixels), 10000, replace=False)
        valid_pixels = valid_pixels[indices]
        valid_normals = valid_normals[indices]
    
    # Compute spherical harmonics basis
    basis = compute_spherical_harmonics_basis(valid_normals, order)
    
    # Solve for lighting coefficients
    model = LinearRegression()
    model.fit(basis, valid_pixels)
    
    return model.coef_

class LightingProcessor:
    """Class to handle lighting coefficient calculation and management."""
    
    def __init__(self, base_path: Path, character_name: str):
        self.base_path = Path(base_path)
        self.character_name = character_name
        self.char_path = self.base_path / "characters" / character_name
    
    def process_frame(self, frame_id: str, face_map_path: Path, normal_map_path: Path):
        """Process a single frame and return lighting coefficients with metadata."""
        try:
            # Read the maps
            face_img = cv2.imread(str(face_map_path))
            normal_map = cv2.imread(str(normal_map_path))
            
            if face_img is None or normal_map is None:
                raise ValueError(f"Could not read images for frame {frame_id}")
            
            # Calculate lighting coefficients
            coeffs = calculate_lighting_coefficients(face_img, normal_map)
            
            # Create frame data
            frame_data = {
                "frame_id": frame_id,
                "lighting_coefficients": coeffs.tolist(),
                "face_map": str(face_map_path.relative_to(self.base_path)),
                "normal_map": str(normal_map_path.relative_to(self.base_path)),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return frame_data
            
        except Exception as e:
            print(f"Error processing lighting for frame {frame_id}: {str(e)}")
            return None
    
    def visualize_lighting(self, coefficients, output_path=None):
        """Generate a visualization of the lighting coefficients."""
        # TODO: Implement visualization (e.g., sphere rendering with lighting)
        pass
