# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing utils."""

import dataclasses
import math
from typing import List, Mapping, Optional, Tuple, Union, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.coordinate_utils as coordinate_utils

try:
  from mediapipe.framework.formats import detection_pb2
  from mediapipe.framework.formats import landmark_pb2
  from mediapipe.framework.formats import location_data_pb2
except ImportError:
    print("Warning: mediapipe not found. Some functionality may be limited.")
    # Create dummy classes for the protobuf messages
    class DummyProto:
        def __init__(self):
            self.landmark = []
            self.location_data = None
            self.HasField = lambda x: False
            
    class DummyLocationData:
        def __init__(self):
            self.RELATIVE_BOUNDING_BOX = 0
            self.format = 0
            self.relative_keypoints = []
            
    detection_pb2 = type('Detection', (), {'Detection': DummyProto})
    landmark_pb2 = type('NormalizedLandmarkList', (), {'NormalizedLandmarkList': DummyProto})
    location_data_pb2 = type('LocationData', (), {'LocationData': DummyLocationData})

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2


def __normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def __get_grayscale_from_depth(z, z_min, z_max):
  """
  Maps a z-value to a grayscale color from black (near) to white (far).
  z_min: the closest depth (smallest z)
  z_max: the farthest depth (largest z)
  Returns a BGR color tuple.
  """
  # Normalize z to range [0, 1]
  normalized_z = (z - z_min) / (z_max - z_min + 1e-6)
  intensity = int(normalized_z * 255)
  return (intensity, intensity, intensity)  # BGR grayscale
  
def __load_obj_vertices_faces(obj_path):
  """Load vertices and faces from an OBJ file with debug information."""
  vertices = []
  vertex_normals = []
  faces = []
  print(f"\nLoading OBJ file: {obj_path}")
  
  try:
    with open(obj_path, 'r') as f:
      vertex_count = 0
      normal_count = 0
      face_count = 0
      for line in f:
          if line.startswith('v '):  # vertex
              try:
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), -float(z)])
                vertex_count += 1
              except ValueError as e:
                print(f"Warning: Invalid vertex line: {line.strip()}")
                continue
          elif line.startswith('vn '):  # vertex normal
              try:
                _, x, y, z = line.strip().split()
                vertex_normals.append([float(x), float(y), -float(z)])
                normal_count += 1
              except ValueError as e:
                print(f"Warning: Invalid normal line: {line.strip()}")
                continue
          elif line.startswith('f '):  # face
              try:
                # Handle different face formats (v, v/vt, v/vt/vn)
                face_indices = []
                normal_indices = []
                for idx in line.strip().split()[1:]:
                  # Split by '/' and get vertex and normal indices
                  parts = idx.split('/')
                  if len(parts) >= 3:  # v/vt/vn format
                    v_idx = parts[0]
                    n_idx = parts[2] if len(parts) > 2 else None
                    if v_idx:
                      face_indices.append(int(v_idx) - 1)  # OBJ indices are 1-based
                    if n_idx:
                      normal_indices.append(int(n_idx) - 1)
                  else:  # v format
                    face_indices.append(int(parts[0]) - 1)
                
                if len(face_indices) >= 3:
                  faces.append(face_indices[:3])  # use triangle only
                  face_count += 1
              except (ValueError, IndexError) as e:
                print(f"Warning: Invalid face line: {line.strip()}")
                continue
        
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    if len(vertices) == 0:
      raise ValueError("No vertices found in OBJ file")
    if len(faces) == 0:
      raise ValueError("No faces found in OBJ file")
      
    return vertices, faces
  except Exception as e:
    print(f"Error loading OBJ file: {str(e)}")
    raise

def create_normal_map(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    visible_faces: np.array,
    alpha: float = 0.3,
    smooth_factor: float = 1.0
):
  """Creates a normal map by determining which face is in the foreground for each pixel.

  Args:
    image: Input image
    vertices: 3D vertices
    faces: Face indices
    face_normals: Face normals
    visible_faces: Indices of visible faces
    alpha: Blending factor for visualization
    smooth_factor: Amount of Gaussian blur to apply to the normal map (0.0 to disable)
  """
  image_rows, image_cols, _ = image.shape

  # Create a depth buffer and normal map
  depth_buffer = np.full((image_rows, image_cols), np.inf)
  normal_map = np.zeros((image_rows, image_cols, 3), dtype=np.uint8)
  
  # Get 2D vertices
  vertices_2d = vertices[:, :2].astype(np.int32)
  # Process visible faces
  for face_idx in visible_faces:
    face = faces[face_idx]
    face_2d = vertices_2d[face]
    # Skip faces outside the image bounds
    if not (np.all(face_2d[:, 0] >= 0) and np.all(face_2d[:, 0] < image_cols) and
            np.all(face_2d[:, 1] >= 0) and np.all(face_2d[:, 1] < image_rows)):
      continue
    
    # Get face normal and convert to color
    normal = face_normals[face_idx]
    normal[2] = -normal[2]
    color = coordinate_utils.xyz_to_rgb(normal.reshape(1, 3))[0].astype(int)
    
    # Find bounding box of the face
    min_x = max(0, np.min(face_2d[:, 0]))
    max_x = min(image_cols - 1, np.max(face_2d[:, 0]))
    min_y = max(0, np.min(face_2d[:, 1]))
    max_y = min(image_rows - 1, np.max(face_2d[:, 1]))
    
    # Depth for this face (average Z)
    depth = np.mean(vertices[face, 2])
    
    # Check each pixel in the bounding box
    for y in range(min_y, max_y + 1):
      for x in range(min_x, max_x + 1):
        # Check if pixel is inside the triangle
        if cv2.pointPolygonTest(face_2d, (x, y), False) >= 0:
          # If this face is closer than any previous face at this pixel, update the depth buffer and normal map
          if depth < depth_buffer[y, x]:
            depth_buffer[y, x] = depth
            normal_map[y, x] = color
  
  # Create a mask of pixels that have a face
  mask = depth_buffer < np.inf
  
  # Apply smoothing if enabled
  if smooth_factor > 0:
    # Convert to float for smoothing
    normal_map_float = normal_map.astype(float)
    
    # Apply Gaussian blur to each channel separately
    for c in range(3):
      normal_map_float[..., c] = cv2.GaussianBlur(
          normal_map_float[..., c],
          (0, 0),  # Let OpenCV calculate kernel size
          sigmaX=smooth_factor,
          sigmaY=smooth_factor
      )
    
    # Convert back to uint8
    normal_map = normal_map_float.astype(np.uint8)
    
    # Ensure the smoothed normals are still normalized
    # Convert to float for normalization
    normal_map_float = normal_map.astype(float)
    # Normalize each pixel
    norm = np.sqrt(np.sum(normal_map_float**2, axis=2, keepdims=True))
    normal_map_float = normal_map_float / (norm + 1e-6)
    # Convert back to uint8
    normal_map = (normal_map_float * 255).astype(np.uint8)
  
  # Create a copy of the image for blending
  overlay = image.copy()
  overlay[mask] = normal_map[mask]
  overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
  cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
  
  return normal_map, mask

def draw_surface_normals(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_coordinates: dict,
    camera_position: np.ndarray = np.array([0, 0, -1]),
    alpha: float = 0.5,
    smooth_factor: float = 1.0
):
  """Draws surface normals as colored faces on the image."""
  if image.shape[2] != _BGR_CHANNELS:
    return None
  
  # Step 1: Calculate face normals
  face_normals = []
  for face in faces:
    v1, v2, v3 = vertices[face]
    face_normal = -np.cross(v2 - v1, v3 - v1)
    face_normal /= np.linalg.norm(face_normal) + 1e-8  # Normalize
    face_normals.append(face_normal)
  face_normals = np.array(face_normals)
  
  # Step 2: Scale vertices to match landmark size
  # Get landmark bounds
  landmark_xs = [coord[0] for coord in landmark_coordinates.values()]
  landmark_ys = [coord[1] for coord in landmark_coordinates.values()]
  landmark_min_x, landmark_max_x = min(landmark_xs), max(landmark_xs)
  landmark_min_y, landmark_max_y = min(landmark_ys), max(landmark_ys)
  landmark_width = landmark_max_x - landmark_min_x
  landmark_height = landmark_max_y - landmark_min_y
  
  # Get model bounds
  min_x, min_y = np.min(vertices[:, :2], axis=0)
  max_x, max_y = np.max(vertices[:, :2], axis=0)
  model_width = max_x - min_x
  model_height = max_y - min_y
  
  # Calculate and apply scaling
  scale_x = landmark_width / (model_width + 1e-8)
  scale_y = landmark_height / (model_height + 1e-8)
  
  vertices_scaled = vertices.copy()
  vertices_scaled[:, 0] = vertices[:, 0] * scale_x
  vertices_scaled[:, 1] = vertices[:, 1] * scale_y
  vertices_scaled[:, 2] = vertices[:, 2] * scale_x
  
  # Step 3: Determine visible faces
  camera_direction = -camera_position / (np.linalg.norm(camera_position) + 1e-8)
  visibility = np.dot(face_normals, camera_direction)
  visible_faces = np.where(visibility > -0.8)[0]
  
  # Step 4: Create normal map
  normal_map, mask = create_normal_map(
      image,
      vertices_scaled,
      faces,
      face_normals,
      visible_faces,
      alpha,
      smooth_factor
  )
  
  return normal_map, mask

def create_blended_normal_map(
    image,
    landmark_list,
    vertices,
    faces,
    obj_path,
    smoothness=0.,
    intensity=1.,
    smooth_factor=1.0,
    debug=False,
    target_size=1024
):
    """
    Create a blended normal map by combining mesh-based and image-based normal maps.
    All outputs are resized to target_size x target_size while maintaining aspect ratio.
    
    Returns:
        tuple: (blended_normal_map, face_square, mask, ao_map)
    """
    # Create a copy of the image to prevent modifying the original
    image_copy = image.copy()
    
    # Create a dictionary mapping landmark indices to coordinates
    image_rows, image_cols, _ = image_copy.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
             landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = __normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    # Create mesh-based normal map
    mesh_normal_map, mesh_mask = draw_surface_normals(
        image=image_copy,
        vertices=vertices,
        faces=faces,
        landmark_coordinates=idx_to_coordinates,
        smooth_factor=smooth_factor
    )
    
    if debug:
        # Save mesh-based normal map for debugging
        cv2.imwrite('debug_mesh_normal_map.png', cv2.cvtColor(mesh_normal_map, cv2.COLOR_RGB2BGR))
        # Debug coordinate conversions
        from coordinate_utils import debug_coordinate_conversions
        debug_coordinate_conversions(mesh_normal_map, mesh_mask)
    
    # Create temporary image file for normal_map_generator with only face pixels
    temp_image_path = 'temp_face_image.png'
    # Create a copy of the image
    face_only_image = image.copy()
    # Set all non-face pixels to black
    face_only_image[~mesh_mask] = [0, 0, 0]
    
    # Get the bounding box of the face mask
    y_indices, x_indices = np.where(mesh_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Calculate the size of the square (use the larger dimension)
    size = max(y_max - y_min, x_max - x_min)
    
    # Create a square image for the face data
    face_square = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Calculate the offset to center the face in the square
    y_offset = (size - (y_max - y_min)) // 2
    x_offset = (size - (x_max - x_min)) // 2
    
    # Copy the face data to the square image
    face_square[y_offset:y_offset + (y_max - y_min), 
                x_offset:x_offset + (x_max - x_min)] = face_only_image[y_min:y_max, x_min:x_max]
    
    # Resize face_square to target size
    face_square = cv2.resize(face_square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    if debug:
        # Save the square face-only image for debugging
        cv2.imwrite('debug_face_only_image.png', face_square)
    
    # Save the square face image as temporary file for normal_map_generator
    cv2.imwrite(temp_image_path, face_square)
    
    # Convert mesh mask to binary format (0 or 255)
    binary_mask = (mesh_mask * 255).astype(np.uint8)
    
    # Create square binary mask and resize to target size
    square_binary_mask = np.zeros((size, size), dtype=np.uint8)
    square_binary_mask[y_offset:y_offset + (y_max - y_min), 
                      x_offset:x_offset + (x_max - x_min)] = binary_mask[y_min:y_max, x_min:x_max]
    square_binary_mask = cv2.resize(square_binary_mask, (target_size, target_size), 
                                  interpolation=cv2.INTER_NEAREST)
    
    # Generate image-based normal map using normal_map_generator
    from utils.normal_map_generator import startConvert
    image_normal_map, ao_map = startConvert(
        input_file=temp_image_path,
        smooth=smoothness,
        intensity=intensity,
        mask=square_binary_mask
    )
    
    if debug:
        # Save image-based normal map for debugging
        cv2.imwrite('debug_image_normal_map.png', image_normal_map)
        cv2.imwrite('debug_ao_map.png', ao_map)
        # Debug coordinate conversions for image-based normal map
        # Convert square_binary_mask to boolean
        square_mask = square_binary_mask.astype(bool)
        debug_coordinate_conversions(image_normal_map, square_mask)
    
    # Create a square mesh normal map
    square_mesh_normal = np.zeros((size, size, 3), dtype=np.uint8)
    square_mesh_normal[y_offset:y_offset + (y_max - y_min), 
                      x_offset:x_offset + (x_max - x_min)] = mesh_normal_map[y_min:y_max, x_min:x_max]
    
    # Resize mesh normal map to target size
    square_mesh_normal = cv2.resize(square_mesh_normal, (target_size, target_size), 
                                  interpolation=cv2.INTER_LINEAR)
    
    if debug:
        cv2.imwrite('debug_square_mesh_normal.png', cv2.cvtColor(square_mesh_normal, cv2.COLOR_RGB2BGR))
    
    # Create blending weights
    mesh_weight = np.mean(np.abs(image_normal_map - 127.5), axis=2) / 127.5
    mesh_weight = np.clip(mesh_weight * 2, 0, 1)  # Amplify the weight
    mesh_weight = np.stack([mesh_weight] * 3, axis=2)
    
    # Blend the normal maps (now they have the same dimensions)
    blended_normal_map = (
        cv2.cvtColor(square_mesh_normal, cv2.COLOR_RGB2BGR) * (1-mesh_weight) +
        image_normal_map * mesh_weight
    ).astype(np.uint8)
    
    # Create a black background
    black_background = np.zeros_like(blended_normal_map)
    
    # Create square mask for the blended normal map
    square_mask = square_binary_mask.astype(bool)
    
    # Apply the mask to keep only the face region
    blended_normal_map = np.where(
        square_mask[..., np.newaxis],  # Expand mask to 3 channels
        blended_normal_map,
        black_background
    )
    
    if debug:
        # Save blended normal map for debugging
        cv2.imwrite('debug_blended_normal_map.png', blended_normal_map)
        # Save the square face image for debugging
        cv2.imwrite('debug_face_square.png', face_square)
        # Debug coordinate conversions for both images
        debug_coordinate_conversions(blended_normal_map, square_mask)
    
    # Clean up temporary files
    import os
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    # Return all the processed maps instead of saving them
    return blended_normal_map, face_square, square_mask, ao_map

def draw_landmarks(
    image: np.ndarray,
    landmark_list: Any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[
        Union[DrawingSpec, Mapping[int, DrawingSpec]]
    ] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
    obj_path: Optional[str] = None
):
  """Draws the landmarks and the connections on the image."""
  if not landmark_list or image.shape[2] != _BGR_CHANNELS:
    return
  
  # Collect landmarks
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  idx_to_z = {}
  
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = __normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
      idx_to_z[idx] = landmark.z

  # Create a copy of landmarks for later use
  visible_landmarks = idx_to_coordinates.copy()
  
  # Draw surface normals if OBJ file is provided
  if obj_path:
    try:
      vertices, faces = __load_obj_vertices_faces(obj_path)
      blended_normal_map, face_image, mask, ao_map = create_blended_normal_map(
          image=image,
          landmark_list=landmark_list,
          vertices=vertices,
          faces=faces,
          obj_path=obj_path,
          debug=False
      )
      
      print(f"\nDebug shapes:")
      print(f"Image shape: {image.shape}")
      print(f"Normal map shape: {blended_normal_map.shape}")
      print(f"Mask shape: {mask.shape}")
      print(f"Number of valid pixels: {np.sum(mask)}")
      
      # Only proceed with lighting estimation if we have valid pixels
      if mask.any():
        try:
          from lighting_utils import estimate_lighting, calculate_lighting_from_normals, visualize_single_sphere
          
          num_bands = 2
          
          # Estimate lighting coefficients
          lighting_coeffs, lighting_map,Y = estimate_lighting(blended_normal_map, mask, num_bands, True)

          # Get single sphere visualization with custom lighting coefficients
          sphere_vis = visualize_single_sphere(lighting_coeffs)
          cv2.imwrite('sphere_visualization.png', sphere_vis)
          
          # Calculate lighting map
          lighting_map = calculate_lighting_from_normals(blended_normal_map, lighting_coeffs, mask, num_bands, Y)
          
          # Save or display the lighting map
          cv2.imwrite('lighting_map.png', lighting_map)
        except ImportError:
          print("Warning: lighting_utils not found. Lighting calculation skipped.")
        except Exception as e:
          print(f"Warning: Failed to calculate lighting map: {e}")
          import traceback
          traceback.print_exc()
      
      # Restore original landmarks
      idx_to_coordinates = visible_landmarks
      
    except Exception as e:
      print(f"Warning: Failed to draw surface normals: {e}")
      import traceback
      traceback.print_exc()

  # Draw connections
  if connections:
      num_landmarks = len(landmark_list.landmark)
      for connection in connections:
          start_idx, end_idx = connection
          if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
              raise ValueError(f'Invalid connection from landmark #{start_idx} to #{end_idx}.')

          if (start_idx in idx_to_coordinates and end_idx in idx_to_coordinates):
              drawing_spec = connection_drawing_spec[connection] if isinstance(
                  connection_drawing_spec, Mapping) else connection_drawing_spec
              cv2.line(
                  image,
                  idx_to_coordinates[start_idx],
                  idx_to_coordinates[end_idx],
                  drawing_spec.color,
                  drawing_spec.thickness,
              )

  # Draw landmarks
  if is_drawing_landmarks and landmark_drawing_spec:
      for idx, landmark_px in idx_to_coordinates.items():
          drawing_spec = landmark_drawing_spec[idx] if isinstance(
              landmark_drawing_spec, Mapping) else landmark_drawing_spec
          grayscale = __get_grayscale_from_depth(idx_to_z[idx],min(idx_to_z.values()),max(idx_to_z.values()))
          circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
          cv2.circle(image, landmark_px, circle_border_radius, grayscale,
                    drawing_spec.thickness)
          cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                    grayscale, drawing_spec.thickness)
