import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import trimesh
import pyvista as pv
import os
import argparse
from pathlib import Path
import utils.drawing_utils as mp_drawing
import utils.opencv_brightness as cvb
import re

REF_IMAGE = path = "C:\\Users\\balag\\ARTISSN\\Swapping\\propio\\swapper\\data\\characters\\documale1\\source\\images\\frame_00200.jpg"

mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def show_mesh(obj_path):
  mesh = pv.read(obj_path)
  mesh.plot()

def vertices2obj(vertices,out_path):
  # Load original .obj file
  with open("utils/face_model_with_iris.obj", "r") as file:
      lines = file.readlines()

  # Convert vertices list to numpy array for rotation
  vertices_array = np.array(vertices)
  
  # First rotate 180 degrees about Z-axis (negate X and Y)
  rotated_vertices = vertices_array.copy()
  rotated_vertices[:, 0] = -vertices_array[:, 0]  # Negate X
  rotated_vertices[:, 1] = -vertices_array[:, 1]  # Negate Y
  # Z remains unchanged as it's the rotation axis
  
  # Then rotate 180 degrees about Y-axis (negate X and Z)
  rotated_vertices[:, 0] = -rotated_vertices[:, 0]  # Negate X again
  rotated_vertices[:, 2] = -rotated_vertices[:, 2]  # Negate Z
  # Y remains unchanged as it's the rotation axis

  # Map our coordinate system to OBJ format
  # Our Y (up) becomes OBJ's -Y (up)
  # Our Z (forward) becomes OBJ's Z (forward)
  # Our X (right) becomes OBJ's X (right)
  obj_vertices = rotated_vertices.copy()
  obj_vertices[:, 1] = -rotated_vertices[:, 1]  # Flip Y to match OBJ convention

  # Translate vertices to ensure all Y values are positive
  min_y = np.min(obj_vertices[:, 1])
  if min_y < 0:
      obj_vertices[:, 1] -= min_y  # Add the absolute value of min_y to all Y coordinates

  # Replace vertex lines
  new_lines = []
  vertex_index = 0

  for line in lines:
      if line.startswith("v "):  # old vertex line
          v = obj_vertices[vertex_index]
          new_line = f"v {v[0]} {v[1]} {v[2]}\n"
          new_lines.append(new_line)
          vertex_index += 1
      else:
          new_lines.append(line)  # keep everything else (faces, comments, etc.)

  # Sanity check
  if vertex_index != len(vertices):
      raise ValueError("Number of new vertices does not match the original OBJ file!")

  # Save the updated .obj file
  with open(out_path, "w") as file:
      file.writelines(new_lines)

  print("OBJ file built successfully!")
  return out_path

def crop_face(image_path, face_mesh, output_dir, target_size=512):
    """Crop face from image, preserve aspect ratio, and pad to fixed size."""
    base_name = Path(image_path).stem
    output_path = os.path.join(output_dir, "processed", "faces", f"{base_name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Fix brightness
    global REF_IMAGE
    #image = cvb.match_histogram_lab(image, cv2.imread(REF_IMAGE))
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Detect face landmarks with image dimensions
    results = face_mesh.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    
    if not results.multi_face_landmarks:
        print(f"No face detected in: {image_path}")
        return None
    
    face_landmarks = results.multi_face_landmarks[0]
    
    # Find face bounds
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for lm in face_landmarks.landmark:
        x, y = lm.x, lm.y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    
    left = int(min_x * w)
    right = int(max_x * w)
    top = int(min_y * h)
    bottom = int(max_y * h)
    
    # Add padding (optional, can be adjusted)
    padding = 100
    top = max(0, top - padding)
    bottom = min(h, bottom + padding)
    left = max(0, left - padding)
    right = min(w, right + padding)
    
    # Crop as a rectangle (no square enforcement)
    cropped_image = image[top:bottom, left:right]
    crop_h, crop_w = cropped_image.shape[:2]
    
    # Compute scaling factor to fit within target_size
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = cv2.resize(cropped_image, (new_w, new_h))
    
    # Create black background and paste resized crop in the center
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    cv2.imwrite(output_path, padded)
    print(f"Cropped image saved to: {output_path}")
    
    return {
        'crop_path': output_path,
        'original_bounds': (top, bottom, left, right),
        'face_landmarks': face_landmarks
    }

def get_face_mesh_indices(landmark_list, image_width, image_height):
    indices = []
    for lm in landmark_list.landmark:
        x_px = min(int(lm.x * image_width), image_width - 1)
        y_px = min(int(lm.y * image_height), image_height - 1)
        indices.append((x_px, y_px))
    return indices

def generate_face_mesh(image_path, character_dir, face_mesh):
    """Generate face mesh from an image, first cropping the face."""
    base_name = Path(image_path).stem
    meshes_dir = os.path.join(character_dir, "processed", "meshes")
    obj_output_path = os.path.join(meshes_dir, f"{base_name}.obj")
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Crop the face first
    crop_result = crop_face(image_path, face_mesh, character_dir)
    if not crop_result:
        print(f"Failed to crop face from: {image_path}")
        return None
    
    # Now process the cropped face
    cropped_image = cv2.imread(crop_result['crop_path'])
    if cropped_image is None:
        print(f"Failed to read cropped image: {crop_result['crop_path']}")
        return None
    
    # Process the cropped face
    results = face_mesh.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print(f"No face detected in cropped image: {crop_result['crop_path']}")
        return None
    
    face_landmarks = results.multi_face_landmarks[0]

    face_mesh_indices = get_face_mesh_indices(face_landmarks, cropped_image.shape[1], cropped_image.shape[0])
    
    # Convert landmarks to vertices
    vertices = []
    for landmark in face_landmarks.landmark:
        vertices.append([landmark.x, landmark.y, landmark.z])
    
    # Generate OBJ file
    obj_path = vertices2obj(vertices, obj_output_path)

    # After face_landmarks = results.multi_face_landmarks[0]
    landmarks_list = []
    for lm in face_landmarks.landmark:
        lm_dict = {"x": lm.x, "y": lm.y, "z": lm.z}
        if hasattr(lm, "visibility"):
            lm_dict["visibility"] = lm.visibility
        if hasattr(lm, "presence"):
            lm_dict["presence"] = lm.presence
        landmarks_list.append(lm_dict)

    return {
        'obj_path': obj_path,
        'landmarks': face_landmarks,
        'landmarks_list': landmarks_list,
        'image': cropped_image,
        'crop_bounds': crop_result['original_bounds'],
        'face_mesh_indices': face_mesh_indices,
        'crop_path': crop_result['crop_path']
    }

def generate_normal_maps(mesh_data, processed_dir):
    """Generate face cutout and normal maps from mesh data."""
    base_name = Path(mesh_data['obj_path']).stem
    face_output_path = os.path.join(processed_dir, "faces", f"{base_name}.png")
    normal_output_path = os.path.join(processed_dir, "normals", f"{base_name}.png")
    ao_output_path = os.path.join(processed_dir, "ao", f"{base_name}_ao.png")
    
    # Draw landmarks and generate normal map
    annotated_image = mesh_data['image'].copy()
    
    # Load vertices and faces from the OBJ file
    vertices, faces = mp_drawing.__load_obj_vertices_faces(mesh_data['obj_path'])
    
    if not mesh_data['landmarks'] or len(mesh_data['landmarks'].landmark) == 0:
        print(f"Warning: No landmarks found for {mesh_data['obj_path']}. Skipping.")
        return None
    
    # Generate the maps
    normal_map, face_image, mask, ao_map = mp_drawing.create_blended_normal_map(
        image=annotated_image,
        landmark_list=mesh_data['landmarks'],
        vertices=vertices,
        faces=faces,
        obj_path=mesh_data['obj_path'],
        debug=False,
        target_size=512
    )
    
    # Save the maps directly
    cv2.imwrite(face_output_path, face_image)
    cv2.imwrite(normal_output_path, normal_map)

    # Uncomment for AO maps
    #if ao_map is not None:
    #    cv2.imwrite(ao_output_path, ao_map)
    
    return {
        'face_path': face_output_path,
        'normal_path': normal_output_path,
        'ao_path': ao_output_path if ao_map is not None else None,
        'mask': mask  # Return the mask in case it's needed later
    }

def extract_number(path):
    match = re.search(r'\d+', path.stem)  # path.stem = filename without extension
    return int(match.group()) if match else -1

def process_directory(input_dir, output_dir):
    """Process all images in the input directory."""
    # Create all necessary directories upfront
    processed_dir = os.path.join(output_dir, "processed")
    faces_dir = os.path.join(processed_dir, "faces")
    meshes_dir = os.path.join(processed_dir, "meshes")
    normals_dir = os.path.join(processed_dir, "normals")
    
    for dir_path in [processed_dir, faces_dir, meshes_dir, normals_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Sort numerically
    image_files = sorted(image_files, key=extract_number)
    
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3) as face_mesh:
        
        # First pass: crop all faces
        cropped_faces = []
        for idx, image_path in enumerate(image_files):
            print(f"Cropping face {idx + 1}/{len(image_files)}: {image_path}")
            try:
                result = crop_face(str(image_path), face_mesh, output_dir)
                if result:
                    cropped_faces.append(result)
            except Exception as e:
                print(f"Error cropping {image_path}: {str(e)}")
                continue
        
        # Second pass: generate meshes from cropped faces
        for face_data in cropped_faces:
            try:
                mesh_data = generate_face_mesh(
                    face_data['crop_path'],
                    meshes_dir,  # Use the specific meshes directory
                    face_mesh
                )
                if mesh_data:
                    # Pass the processed directory for normal maps
                    generate_normal_maps(mesh_data, processed_dir)
            except Exception as e:
                print(f"Error processing {face_data['crop_path']}: {str(e)}")
                import traceback
                traceback.print_exc()  # Add this to get more detailed error information
                continue

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process face images to generate normal maps')
    parser.add_argument('--input_dir', type=str, default='input_faces',
                        help='Directory containing input face images')
    parser.add_argument('--output_dir', type=str, default='output_faces',
                        help='Directory to save processed faces and normal maps')
    args = parser.parse_args()
    
    # Process the directory
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
