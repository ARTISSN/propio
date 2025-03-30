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
            vertices.append([float(x), float(y), float(z)])
            vertex_count += 1
          except ValueError as e:
            print(f"Warning: Invalid vertex line: {line.strip()}")
            continue
        elif line.startswith('vn '):  # vertex normal
          try:
            _, x, y, z = line.strip().split()
            vertex_normals.append([float(x), float(y), float(z)])
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
    
    print(f"Successfully loaded OBJ file:")
    print(f"- Number of vertices: {len(vertices)}")
    print(f"- Number of vertex normals: {len(vertex_normals)}")
    print(f"- Number of faces: {len(faces)}")
    print(f"- Vertex shape: {vertices.shape}")
    print(f"- Face shape: {faces.shape}")
    
    if len(vertices) == 0:
      raise ValueError("No vertices found in OBJ file")
    if len(faces) == 0:
      raise ValueError("No faces found in OBJ file")
      
    return vertices, faces
    
  except FileNotFoundError:
    print(f"Error: OBJ file not found: {obj_path}")
    raise
  except Exception as e:
    print(f"Error loading OBJ file: {str(e)}")
    raise

def __normalize_color(color):
  return tuple(v / 255. for v in color)

def __normal_to_color(normal):
  """Convert a 3D normal vector to a BGR color.
  Maps the normal direction to RGB color space where:
  - Red represents X component
  - Green represents Y component
  - Blue represents Z component
  """
  # Map normal from [-1,1] to [0,1] range
  color = (normal + 1) / 2
  # Convert to BGR (OpenCV uses BGR)
  return (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

def create_normal_map(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    visible_faces: np.array,
    alpha: float = 0.3
):
  """Creates a normal map by determining which face is in the foreground for each pixel."""
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
    color = __normal_to_color(normal)
    
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
  
  # Create a copy of the image for blending
  overlay = image.copy()
  overlay[mask] = normal_map[mask]
  cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
  
  return normal_map, mask

def draw_surface_normals(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_coordinates: dict,
    camera_position: np.ndarray = np.array([0, 0, 1]),
    alpha: float = 0.5
):
  """Draws surface normals as colored faces on the image."""
  if image.shape[2] != _BGR_CHANNELS:
    return None
  
  # Step 1: Calculate face normals
  face_normals = []
  for face in faces:
    v1, v2, v3 = vertices[face]
    face_normal = -np.cross(v2 - v1, v3 - v1)  # Invert normal
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
  camera_direction = camera_position / (np.linalg.norm(camera_position) + 1e-8)
  visibility = np.dot(face_normals, camera_direction)
  visible_faces = np.where(visibility > -0.8)[0]
  
  # Step 4: Create normal map
  normal_map, mask = create_normal_map(image, vertices_scaled, faces, face_normals, visible_faces, alpha)
  
  return normal_map, mask

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
      normal_map, mask = draw_surface_normals(image, vertices, faces, idx_to_coordinates)
      
      print(f"\nDebug shapes:")
      print(f"Image shape: {image.shape}")
      print(f"Normal map shape: {normal_map.shape}")
      print(f"Mask shape: {mask.shape}")
      print(f"Number of valid pixels: {np.sum(mask)}")
      
      # Only proceed with lighting estimation if we have valid pixels
      if mask.any():
        try:
          from lighting_utils import estimate_lighting, calculate_lighting_from_normals
          
          # Estimate lighting coefficients
          lighting_coeffs = estimate_lighting(normal_map, mask)
          
          # Calculate lighting map
          lighting_map = calculate_lighting_from_normals(normal_map, lighting_coeffs, mask)
          
          # Save or display the lighting map
          cv2.imwrite('lighting_map.png', lighting_map)
          
          # Optionally, create a colored visualization
          lighting_rgb = cv2.cvtColor(lighting_map, cv2.COLOR_GRAY2BGR)
          cv2.imwrite('lighting_map_colored.png', lighting_rgb)
          
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

def plot_landmarks(landmark_list: Any,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
  """Plot the landmarks and the connections in matplotlib 3d.

  Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.
  """
  if not landmark_list:
    return
  plt.figure(figsize=(10, 10))
  ax = plt.axes(projection='3d')
  ax.view_init(elev=elevation, azim=azimuth)
  
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    ax.scatter3D(
        xs=[-landmark.z],
        ys=[landmark.x],
        zs=[-landmark.y],
        color=__normalize_color(landmark_drawing_spec.color[::-1]),
        linewidth=landmark_drawing_spec.thickness)
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

  if connections:
    num_landmarks = len(landmark_list.landmark)
    for connection in connections:
      start_idx, end_idx = connection
      if (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks and
          start_idx in plotted_landmarks and end_idx in plotted_landmarks):
        landmark_pair = [plotted_landmarks[start_idx], plotted_landmarks[end_idx]]
        ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color=__normalize_color(connection_drawing_spec.color[::-1]),
            linewidth=connection_drawing_spec.thickness)
  plt.show()
