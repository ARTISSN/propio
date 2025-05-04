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

#import bpy
#import bmesh
#import mathutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.coordinate_utils as coordinate_utils
import os
import trimesh

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

def load_material_colors(mtl_path):
    """Parse MTL file and return a dict of material name to Kd color."""
    colors = {}
    current = None
    with open(mtl_path, 'r') as f:
        for line in f:
            if line.startswith('newmtl'):
                current = line.strip().split()[1]
            elif line.startswith('Kd') and current:
                kd = [float(x) for x in line.strip().split()[1:4]]
                colors[current] = kd
    return colors

def load_face_materials(obj_path):
    """Parse OBJ file and return a list of material names per face (same order as faces)."""
    face_materials = []
    current_mat = None
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('usemtl'):
                current_mat = line.strip().split()[1]
            elif line.startswith('f '):
                face_materials.append(current_mat)
    return face_materials

def load_face_vertex_material_map(obj_path):
    """
    Returns a dict mapping sorted vertex index tuples to material names.
    """
    face_vertex_to_material = {}
    current_mat = None
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('usemtl'):
                current_mat = line.strip().split()[1]
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                v_indices = []
                for idx in parts:
                    v_idx = int(idx.split('/')[0]) - 1  # OBJ is 1-based
                    v_indices.append(v_idx)
                key = tuple(sorted(v_indices))
                face_vertex_to_material[key] = current_mat
    return face_vertex_to_material

def create_normal_map(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    visible_faces: np.array,
    obj_path: str = None,
    mtl_path: str = None,
    alpha: float = 0.3,
    smooth_factor: float = 0.5
):
    image_rows, image_cols, _ = image.shape

    # Create a depth buffer, normal map, and albedo map
    depth_buffer = np.full((image_rows, image_cols), np.inf)
    normal_map = np.zeros((image_rows, image_cols, 3), dtype=np.uint8)
    albedo_map = np.zeros((image_rows, image_cols, 3), dtype=np.uint8)

    # Prepare material info if available
    face_materials = load_face_materials(obj_path) if obj_path and mtl_path else None
    material_colors = load_material_colors(mtl_path) if obj_path and mtl_path else None
    face_vertex_to_material = load_face_vertex_material_map(obj_path) if obj_path and mtl_path else {}

    vertices_2d = vertices[:, :2].astype(np.int32)
    iris_vertices_2d = []
    eye_faces_2d = []
    print("FACE NORMALS: ", face_normals.shape[0])
    for face_idx in visible_faces:
        face = faces[face_idx]
        face_2d = vertices_2d[face]
        if not (np.all(face_2d[:, 0] >= 0) and np.all(face_2d[:, 0] < image_cols) and
                np.all(face_2d[:, 1] >= 0) and np.all(face_2d[:, 1] < image_rows)):
            continue

        normal = face_normals[face_idx]
        normal[2] = -normal[2]
        color = coordinate_utils.xyz_to_rgb(normal.reshape(1, 3))[0].astype(int)

        # Get the material by matching sorted vertex indices
        key = tuple(sorted(face))
        mat_name = face_vertex_to_material.get(key, None)
        if mat_name == 'iris':
            iris_vertices_2d.extend(face_2d.tolist())
            continue  # Skip iris faces
        elif mat_name == 'eye':
            eye_faces_2d.append(face_2d)
        if mat_name and material_colors:
            kd = material_colors.get(mat_name, [0.8, 0.8, 0.8])
            kd_rgb = (np.array(kd) * 255).astype(np.uint8)
        else:
            kd_rgb = np.array([204, 204, 204], dtype=np.uint8)

        min_x = max(0, np.min(face_2d[:, 0]))
        max_x = min(image_cols - 1, np.max(face_2d[:, 0]))
        min_y = max(0, np.min(face_2d[:, 1]))
        max_y = min(image_rows - 1, np.max(face_2d[:, 1]))
        depth = np.mean(vertices[face, 2])

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if cv2.pointPolygonTest(face_2d, (x, y), False) >= 0:
                    if depth < depth_buffer[y, x]:
                        depth_buffer[y, x] = depth
                        normal_map[y, x] = color
                        albedo_map[y, x] = kd_rgb[::-1]

    mask = depth_buffer < np.inf
    
    # 2. If we have iris vertices, compute bounding box, center, and radius
    if iris_vertices_2d:
        iris_vertices_2d = np.array(iris_vertices_2d)
        x_median = np.median(iris_vertices_2d[:, 0])
        left_cluster = iris_vertices_2d[iris_vertices_2d[:, 0] < x_median]
        right_cluster = iris_vertices_2d[iris_vertices_2d[:, 0] >= x_median]
        # Then repeat the bounding box/circle logic for each cluster

        # 3. Draw a black circle for each iris cluster, clipped to the eye region
        for cluster in [left_cluster, right_cluster]:
            if len(cluster) == 0:
                continue
            # Create an eye mask for this cluster
            eye_mask = np.zeros(albedo_map.shape[:2], dtype=np.uint8)
            for face_2d in eye_faces_2d:
                cv2.fillConvexPoly(eye_mask, face_2d, 255)
            # Draw the iris circle on a separate mask
            circle_mask = np.zeros(albedo_map.shape[:2], dtype=np.uint8)
            center_x = int(np.median(cluster[:, 0]))
            center_y = int(np.median(cluster[:, 1]))
            radius = int(
                max(cluster[:, 0].max() - cluster[:, 0].min(),
                    cluster[:, 1].max() - cluster[:, 1].min()) / 2
            )
            cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
            # Combine masks: only keep circle where it overlaps with the eye
            iris_mask = cv2.bitwise_and(circle_mask, eye_mask)
            # Apply to albedo_map (set to black where iris_mask is 255)
            albedo_map[iris_mask == 255] = (0.2, 0.2, 0.2)

    # Smoothing (if needed) -- apply to both maps
    if smooth_factor > 0:
        for arr in [normal_map, albedo_map]:
            arr_float = arr.astype(float)
            for c in range(3):
                arr_float[..., c] = cv2.GaussianBlur(
                    arr_float[..., c], (0, 0), sigmaX=smooth_factor, sigmaY=smooth_factor
                )
            arr[:] = arr_float.astype(np.uint8)

    # Blending for visualization (unchanged)
    overlay = image.copy()
    overlay[mask] = normal_map[mask]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return normal_map, mask, albedo_map

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
    return None, None
  
  # Step 1: Calculate face normals
  face_normals = []
  print("VERTICES: ", vertices.shape[0])
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
  if not landmark_xs or not landmark_ys:
    print("Warning: No valid landmarks provided for normal map generation. Skipping frame.")
    return None, None
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
  normal_map, mask, albedo_map = create_normal_map(
      image,
      vertices_scaled,
      faces,
      face_normals,
      visible_faces,
      #obj_path="C:\\Users\\balag\\ARTISSN\\Swapping\\propio\\swapper\\utils\\material.obj",
      #mtl_path="C:\\Users\\balag\\ARTISSN\\Swapping\\propio\\swapper\\utils\\material.mtl"
      obj_path=os.path.dirname(os.path.abspath(__file__)) + "/material.obj",
      mtl_path=os.path.dirname(os.path.abspath(__file__)) + "/material.mtl"

  )
  
  return normal_map, mask, albedo_map

def create_blended_normal_map(
    image,
    landmark_list,
    vertices,
    faces,
    obj_path,
    smoothness=0.,
    intensity=1.,
    smooth_factor=1.0,
    debug=True,
    target_size_x=1024,
    target_size_y=1024,
    target_size=512
):
    """
    Create a blended normal map by combining mesh-based and image-based normal maps.
    Now returns maps at the original image size.
    """
    # Create a copy of the image to prevent modifying the original
    image_copy = image.copy()
    
    # Create a dictionary mapping landmark indices to coordinates
    image_rows, image_cols, _ = image_copy.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < 0) or
          (landmark.HasField('presence') and landmark.presence < 0)):
          continue
        landmark_px = __normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    # Create mesh-based normal map (full image size)
    mesh_normal_map, mesh_mask, albedo_map = draw_surface_normals(
        image=image_copy,
        vertices=vertices,
        faces=faces,
        landmark_coordinates=idx_to_coordinates,
        smooth_factor=smooth_factor
    )
    
    if debug:
        cv2.imwrite('debug_mesh_normal_map.png', cv2.cvtColor(mesh_normal_map, cv2.COLOR_RGB2BGR))
        cv2.imwrite('debug_albedo_map.png', albedo_map)
        cv2.imwrite('debug_mesh_mask.png', (mesh_mask.astype(np.uint8) * 255))
    
    # Create a masked version of the original image
    masked_image = image.copy()
    masked_image[mesh_mask == 0] = 0  # Set pixels outside mask to black
    
    # Return the masked image instead of the mask
    return mesh_normal_map, masked_image, albedo_map

def draw_landmarks(
    target_size_x,
    target_size_y,
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
        return None
    
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

    # Return blended_normal_map if created, otherwise return None
    return blended_normal_map if blended_normal_map is not None else None
