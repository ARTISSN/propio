import sys
import os
# Dynamically get the folder where mesh_utils.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add that folder to sys.path
if script_dir not in sys.path:
    sys.path.append(script_dir)

import cv2
import json
import mediapipe as mp
import numpy as np
import trimesh
import pyvista as pv
import argparse
import opencv_brightness as cvb
from pathlib import Path
import drawing_utils as mp_drawing
import bpy

mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def show_mesh(obj_path):
  mesh = pv.read(obj_path)
  mesh.plot()

prev_obj = None

def vertices2obj(vertices,out_path):
  # Load original .obj file
  with open("face_model_with_iris.obj", "r") as file:
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
  #obj_vertices[:, 1] = -rotated_vertices[:, 1]  # Flip Y to match OBJ convention


  # Translate vertices to ensure all Y values are positive
#   min_y = np.min(obj_vertices[:, 1])
#   if min_y < 0:
#       obj_vertices[:, 1] -= min_y  # Add the absolute value of min_y to all Y coordinates

#   #take avg of frame with last frame
#   global prev_obj
#   if prev_obj is not None:
#       obj_vertices = (0.1) * obj_vertices + (0.9) * prev_obj

#   prev_obj = obj_vertices
      
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

frame_transformations = []
prev_face = None
prev_face_crop = None


def process_image(image_path, ref_brightness, output_dir, face_mesh, idx, light_dir):
    """Process a single image and save its face cutout and normal map."""
    # Create output paths
    base_name = Path(image_path).stem
    face_output_path = os.path.join(output_dir, f"{base_name}.png")
    normal_output_path = os.path.join(output_dir, f"{base_name}_normal.png")
    obj_output_path = os.path.join(output_dir, f"{base_name}.obj")
    
    # Read and process image
    image = cv2.imread(image_path)
    #reference image
    ref_image = cv2.imread(ref_brightness)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
        
    """PREPROCESSING HERE
    - brighten image
    - crop image to face (call face_mesh.process(image) x2)
    """
    
    #SAVE PATH IS JUST FOR VIEWING THE IMAGES AFTER THEY'VE BEEN BRIGHTENED
    #YOU CAN PASS THROUGH THE HISTOGRAM FUNCTION IF YOU WANT TO SAVE THE IMAGES ON YOUR COMPUTER BUT YOU DON'T HAVE TO
    #save_path = '/Users/rainergardner-olesen/Desktop/Artissn/face_swap/brightness/test_img_brigthness/' + base_name
    #image = cvb.match_histogram_lab(image, ref_image)

    # Convert the BGR image to RGB before processing
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in: {image_path}")
        return None

    # Process the first face found
    face_landmarks_crop = results.multi_face_landmarks[0]

    # global prev_face_crop
    # if prev_face_crop is not None:
    #     for p in range (len(face_landmarks_crop.landmark)):
    #         face_landmarks_crop.landmark[p].x = (0.2 * face_landmarks_crop.landmark[p].x) + (0.8 * prev_face_crop.landmark[p].x)
    #         face_landmarks_crop.landmark[p].y = (0.2 * face_landmarks_crop.landmark[p].y) + (0.8 * prev_face_crop.landmark[p].y)
    #         face_landmarks_crop.landmark[p].z = (0.2 * face_landmarks_crop.landmark[p].z) + (0.8 * prev_face_crop.landmark[p].z)

    # prev_face_crop = face_landmarks_crop


    #find the max x and y of the landmarks for bounding and cropping
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for lm in face_landmarks_crop.landmark:
        x = lm.x
        y = lm.y

        if x < min_x:
            min_x = x
            leftmost = lm
        if x > max_x:
            max_x = x
            rightmost = lm
        if y < min_y:
            min_y = y
            topmost = lm
        if y > max_y:
            max_y = y
            bottommost = lm

    #since mediapipe coordinates are normalized between 0 and 1, we need to multiply by width and height to find pixels
    h = image.shape[0]
    w = image.shape[1]

    left = int(leftmost.x * w)
    right = int(rightmost.x * w)
    top = int(topmost.y * h)
    bottom = int(bottommost.y * h)

    x_size = right - left
    y_size = bottom - top
    dim_diff = x_size - y_size

    #x_size is larger
    if dim_diff > 0:
        #increase top and bottom
        top -= int(np.floor(dim_diff/2))
        bottom += int(np.ceil(dim_diff/2))
    #y_size is larger
    else:
        #increase left and right
        left += int(np.floor(dim_diff/2))
        right -= int(np.ceil(dim_diff/2))


    #now we crop it with padding which can be whatever we want
    padding = 0
    cropped_image = image[top - padding:bottom + padding, left - padding:right+padding]
    global frame_transformations
    #these are the values you want to subtract from the x and y values of each pixel
    frame_transformations[idx] = [left - padding, top - padding]

    # for r in range (cropped_image.shape[0]):
    #     for c in range (cropped_image.shape[1]):
    #         image[r + frame_trans[idx][1]][c + frame_trans[idx][0]] = 0
    
    save_path = '/Users/rainergardner-olesen/Desktop/Artissn/face_swap/brightness/test_crop_daniel/' + base_name
    #if you want to look at the cropped images
    print("SAVING IMAGE TEST CROP")
    cv2.imwrite(save_path + '.png', cropped_image)

    #now we have the cropped image so just do face detection again
    # Convert the BGR image to RGB before processing
    results = face_mesh.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in: {image_path} cropped")
        return None

    # Process the first face found
    face_landmarks = results.multi_face_landmarks[0]

    #take average of current face and previous one
    # global prev_face
    # if prev_face is not None:
    #     for p in range (len(face_landmarks.landmark)):
    #         face_landmarks.landmark[p].x = (0.2 * face_landmarks.landmark[p].x) + (0.8 * prev_face.landmark[p].x)
    #         face_landmarks.landmark[p].y = (0.2 * face_landmarks.landmark[p].y) + (0.8 * prev_face.landmark[p].y)
    #         face_landmarks.landmark[p].z = (0.2 * face_landmarks.landmark[p].z) + (0.8 * prev_face.landmark[p].z)

    # prev_face = face_landmarks

    # Convert landmarks to vertices
    vertices = []
    #we want to use an offset since 0,0 represnets the top left corner of the image but 0,0,0 represents the center of 3d space in blender
    for landmark in face_landmarks.landmark:
        vertices.append([landmark.x - 0.5, landmark.y - 0.5, landmark.z])
    
    # Generate OBJ file
    obj_path = vertices2obj(vertices, obj_output_path)
    
    # Draw landmarks and generate normal map
    annotated_image = cropped_image.copy()
    normal_map = mp_drawing.draw_landmarks(
        target_size_x=cropped_image.shape[1],
        target_size_y=cropped_image.shape[0],
        image=annotated_image,
        is_drawing_landmarks=False,
        landmark_list=face_landmarks,
        connections=None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
        connection_drawing_spec=None,
        obj_path=obj_path,
        light_dir = light_dir)
    
    #normal_map = cv2.resize(normal_map, (cropped_image.shape[0], cropped_image.shape[1]), interpolation=cv2.INTER_LINEAR)
    normal_map = cv2.resize(normal_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    print("RESIZED")
    for r in range (normal_map.shape[0]):
        for c in range (normal_map.shape[1]):
            if normal_map[r][c].all(0):
                image[r + frame_transformations[idx][1]][c + frame_transformations[idx][0]] = normal_map[r][c]
    
    save_path = '/Users/rainergardner-olesen/Desktop/Artissn/face_swap/brightness/daniel_test/' + base_name
    #if you want to look at the cropped images
    print("SAVING IMAGE PUT BACK")
    cv2.imwrite(save_path + '.png', normal_map)
    
    # The face cutout and normal map are saved inside draw_landmarks
    # We should rename them to match our naming convention
    if os.path.exists('debug_face_only_image.png'):
        os.rename('debug_face_only_image.png', face_output_path)
    if os.path.exists('debug_blended_normal_map.png'):
        os.rename('debug_blended_normal_map.png', normal_output_path)
    
    return True

import re

def extract_number(path):
    match = re.search(r'\d+', path.stem)  # path.stem = filename without extension
    return int(match.group()) if match else -1

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
    
    # Sort the list numerically using extract_number
    image_files = sorted(image_files, key=extract_number)

    #json file for lighting info
    with open('/Users/rainergardner-olesen/Desktop/Artissn/face_swap/propio/metadata.json', 'r') as file:
        lighting_data = json.load(file)
    


    #reference image at full brightness (frame 200)
    refer_img = image_files[3]
    global frame_transformations
    #initialize list of frame transformations for each frame
    frame_transformations = [[] for _ in range(len(image_files))]
    for frame in frame_transformations:
        frame = [0,0]

    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3) as face_mesh:
        
        # Process each image
        for idx, image_path in enumerate(image_files):
            (f"Processing image {idx + 1}/{len(image_files)}: {image_path}")
            try:
                # Get the filename without directory
                filename = os.path.basename(image_path)
                # Remove the extension
                root_name = os.path.splitext(filename)[0] 

                light_dir = lighting_data["frames"][root_name]["lighting"]["sun"]
                process_image(str(image_path), str(refer_img), output_dir, face_mesh, idx, light_dir)
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
    
    # Only parse args after "--"
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)
    
    # Process the directory
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

#process_directory('/Users/rainergardner-olesen/Desktop/Artissn/face_swap/propio/data/characters/documale1/source/images', '/Users/rainergardner-olesen/Desktop/Artissn/face_swap/propio/utils/output_faces')