# scripts/extract_embeddings.py

import os
import json
import numpy as np
import cv2
import open3d as o3d
from utils.embedding_utils import get_face_embedding
from utils.image_utils import load_image

# === CONFIG ===
IMG_DIR = "data/images"
EMBED_DIR = "embeddings/faces"
NORMAL_OUT_DIR = "data/normal_maps"
LIGHTING_OUT_DIR = "data/lighting_coeffs"
MESH_DIR = "data/meshes"

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(NORMAL_OUT_DIR, exist_ok=True)
os.makedirs(LIGHTING_OUT_DIR, exist_ok=True)

def generate_normal_map(mesh_path, out_path, resolution=(512, 512)):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    vis = o3d.visualization.rendering.OffscreenRenderer(*resolution)
    vis.scene.set_background([0, 0, 0, 0])
    vis.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())
    vis.setup_camera(60.0, mesh.get_center() + [0, 0, 1.5], mesh.get_center(), [0, 1, 0])
    normal_img = vis.render_to_normal_image()
    normal_np = np.asarray(normal_img) / 255.0
    normal_rgb = (normal_np + 1.0) / 2.0
    cv2.imwrite(out_path, (normal_rgb * 255).astype(np.uint8))

def estimate_lighting(image_path, out_path):
    # Simplified spherical harmonics estimation placeholder
    image = load_image(image_path, target_size=(128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    sh_coeffs = np.random.randn(9) * 0.1  # Replace with real SH fit if needed
    np.save(out_path, sh_coeffs)

def process_all():
    for fname in os.listdir(IMG_DIR):
        if not fname.endswith(".jpg"):
            continue
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, fname)
        embedding = get_face_embedding(img_path)
        if embedding is not None:
            json.dump(embedding.tolist(), open(os.path.join(EMBED_DIR, fname + ".json"), "w"))
        mesh_path = os.path.join(MESH_DIR, name + ".obj")
        normal_path = os.path.join(NORMAL_OUT_DIR, name + ".png")
        lighting_path = os.path.join(LIGHTING_OUT_DIR, name + ".npy")
        if os.path.exists(mesh_path):
            generate_normal_map(mesh_path, normal_path)
        estimate_lighting(img_path, lighting_path)

if __name__ == "__main__":
    process_all()
