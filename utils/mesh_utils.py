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

def generate_face_mesh(image_path, output_dir, face_mesh):
    """Generate face mesh from an image and save as OBJ."""
    # Create output paths
    base_name = Path(image_path).stem
    obj_output_path = os.path.join(output_dir, f"{base_name}.obj")
    
    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
        

        
    # Convert the BGR image to RGB before processing
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in: {image_path}")
        return None

    # Process the first face found
    face_landmarks = results.multi_face_landmarks[0]
    
    # Convert landmarks to vertices
    vertices = []
    for landmark in face_landmarks.landmark:
        vertices.append([landmark.x, landmark.y, landmark.z])
    
    # Generate OBJ file
    obj_path = vertices2obj(vertices, obj_output_path)
    
    return {
        'obj_path': obj_path,
        'landmarks': face_landmarks,
        'image': image
    }

def generate_normal_maps(mesh_data, output_dir):
    """Generate face cutout and normal maps from mesh data."""
    base_name = Path(mesh_data['obj_path']).stem
    face_output_path = os.path.join(f"{output_dir}/faces", f"{base_name}.png")
    normal_output_path = os.path.join(f"{output_dir}/normals", f"{base_name}.png")
    ao_output_path = os.path.join(f"{output_dir}/ao", f"{base_name}_ao.png")
    
    # Draw landmarks and generate normal map
    annotated_image = mesh_data['image'].copy()
    
    # Load vertices and faces from the OBJ file
    vertices, faces = mp_drawing.__load_obj_vertices_faces(mesh_data['obj_path'])
    
    # Generate the maps
    normal_map, face_image, mask, ao_map = mp_drawing.create_blended_normal_map(
        image=annotated_image,
        landmark_list=mesh_data['landmarks'],
        vertices=vertices,
        faces=faces,
        obj_path=mesh_data['obj_path'],
        debug=False  # Set to True only when debugging
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

def process_image(image_path, output_dir, face_mesh):
    """Process a single image and save its face cutout and normal map."""
    # Create output paths
    base_name = Path(image_path).stem
    face_output_path = os.path.join(output_dir, f"{base_name}.png")
    normal_output_path = os.path.join(output_dir, f"{base_name}_normal.png")
    obj_output_path = os.path.join(output_dir, f"{base_name}.obj")
    
    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
        
    """PREPROCESSING HERE
    - brighten image
    - crop image to face (call face_mesh.process(image) x2)
    """

    # Convert the BGR image to RGB before processing
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in: {image_path}")
        return None

    # Process the first face found
    face_landmarks = results.multi_face_landmarks[0]
    
    # Convert landmarks to vertices
    vertices = []
    for landmark in face_landmarks.landmark:
        vertices.append([landmark.x, landmark.y, landmark.z])
    
    # Generate OBJ file
    obj_path = vertices2obj(vertices, obj_output_path)
    
    # Draw landmarks and generate normal map
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        is_drawing_landmarks=False,
        landmark_list=face_landmarks,
        connections=None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
        connection_drawing_spec=None,
        obj_path=obj_path)
    
    # The face cutout and normal map are saved inside draw_landmarks
    # We should rename them to match our naming convention
    if os.path.exists('debug_face_only_image.png'):
        os.rename('debug_face_only_image.png', face_output_path)
    if os.path.exists('debug_blended_normal_map.png'):
        os.rename('debug_blended_normal_map.png', normal_output_path)
    
    return True

def process_directory(input_dir, output_dir):
    """Process all images in the input directory and save results to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Process each image
        for idx, image_path in enumerate(image_files):
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_path}")
            try:
                process_image(str(image_path), output_dir, face_mesh)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
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