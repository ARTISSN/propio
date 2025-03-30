import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import trimesh
import pyvista as pv
import drawing_utils as mp_drawing

mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def show_mesh(obj_path):
  mesh = pv.read(obj_path)
  mesh.plot()

def vertices2obj(vertices,out_path):
  # Load original .obj file
  with open("face_model_with_iris.obj", "r") as file:
      lines = file.readlines()

  # Replace vertex lines
  new_lines = []
  vertex_index = 0

  for line in lines:
      if line.startswith("v "):  # old vertex line
          v = vertices[vertex_index]
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

def build_mesh(images):
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  image_faces = []
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(images):
      face_vertices = []
      image = cv2.imread(file)
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        continue
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        vertices = []
        for landmark in face_landmarks.landmark:
          vertices.append([landmark.x, landmark.y, landmark.z])
          
          
        obj_path = vertices2obj(vertices,"output.obj") # change for multiple faces
        # tesselation is points
        # contour is outline
        # irises is eyes
        mp_drawing.draw_landmarks(
            image=annotated_image,
            is_drawing_landmarks=False,
            landmark_list=face_landmarks,
            connections=None,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=None,
            obj_path=obj_path)
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())
        # mp_drawing.plot_landmarks(
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=mp_drawing.DrawingSpec(
        #         color=(0, 0, 255), thickness=2),
        #     connection_drawing_spec=mp_drawing.DrawingSpec(
        #         color=(0, 0, 0), thickness=2),
        #     elevation=0,
        #     azimuth=0
        # )
        face_vertices.append(vertices)
      image_faces.append(face_vertices)
      cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)
  return image_faces
    
IMAGE_FILES = ["frames_output/frame_00090.jpg"]

image_faces = build_mesh(IMAGE_FILES)