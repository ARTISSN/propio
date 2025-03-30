import open3d as o3d
import numpy as np

# Create PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

# Estimate normals (optional, improves results)
pcd.estimate_normals()

# Poisson surface reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# mesh.vertices, mesh.triangles now contain the mesh
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
