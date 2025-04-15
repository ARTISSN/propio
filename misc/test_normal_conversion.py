import numpy as np
import cv2
from lighting_utils import calculate_lighting_from_normals

def test_normal_conversion():
    # Create a test normal map (100x100x3)
    H, W = 100, 100
    normal_map = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    
    # Create a more varied test pattern
    for y in range(H):
        for x in range(W):
            # Create smooth gradients for each component
            nx = np.sin(x * np.pi / W)  # Varies -1 to 1
            ny = np.sin(y * np.pi / H)  # Varies -1 to 1
            nz = np.cos(np.sqrt((x/W)**2 + (y/H)**2) * np.pi)  # Radial pattern
            
            # Convert to RGB (0-255)
            normal_map[y, x] = [
                ((nx + 1) * 127.5),
                ((ny + 1) * 127.5),
                ((nz + 1) * 127.5)
            ]
            mask[y, x] = True
    
    print("\n1. Initial normal map:")
    print("Shape:", normal_map.shape)
    print("Value range:", np.min(normal_map), np.max(normal_map))
    
    # Convert to normalized vectors
    normals = normal_map / 255.0
    print("\n2. After /255 normalization:")
    print("Value range:", np.min(normals), np.max(normals))
    
    # Convert to [-1,1] range
    normals = 2.0 * normals - 1.0
    print("\n3. After converting to [-1,1]:")
    print("Value range:", np.min(normals), np.max(normals))
    
    # Extract valid normals
    valid_normals = normals[mask]
    print("\n4. Valid normals:")
    print("Shape:", valid_normals.shape)
    print("Sample vector:", valid_normals[0])
    
    # Normalize vectors
    norms = np.linalg.norm(valid_normals, axis=1, keepdims=True)
    valid_normals = valid_normals / (norms + 1e-8)
    print("\n5. After normalizing to unit vectors:")
    print("Sample vector magnitude:", np.linalg.norm(valid_normals[0]))
    print("Sample vector:", valid_normals[0])
    
    # Convert to spherical coordinates
    phi = np.arccos(valid_normals[:, 2])  # Polar angle (0 to π)
    theta = np.arctan2(valid_normals[:, 1], valid_normals[:, 0])  # Azimuthal angle (0 to 2π)
    print("\n6. Spherical coordinates:")
    print("Phi range:", np.min(phi), np.max(phi))
    print("Theta range:", np.min(theta), np.max(theta))
    
    # Test specific vectors
    print("\n7. Test cases:")
    test_vectors = [
        ("Up", np.array([0, 0, 1])),
        ("Right", np.array([1, 0, 0])),
        ("Forward", np.array([0, 1, 0])),
        ("Diagonal", np.array([1, 1, 1])/np.sqrt(3))
    ]
    
    for name, vector in test_vectors:
        test_phi = np.arccos(vector[2])
        test_theta = np.arctan2(vector[1], vector[0])
        print(f"{name} vector: phi={test_phi:.3f}, theta={test_theta:.3f}")
    
    # Verify conversion is reversible
    print("\n8. Verifying reversibility:")
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    reconstructed_normals = np.stack([x, y, z], axis=1)
    
    error = np.mean(np.abs(reconstructed_normals - valid_normals))
    print("Reconstruction error:", error)
    
    # Sample random points
    print("\n9. Random samples:")
    indices = np.random.choice(len(valid_normals), 5)
    for idx in indices:
        print(f"\nSample {idx}:")
        print("Original normal:", valid_normals[idx])
        print("Spherical coords (phi, theta):", phi[idx], theta[idx])
        print("Magnitude:", np.linalg.norm(valid_normals[idx]))
    
    # Save visualization
    print("\n10. Saving visualization...")
    # Create a colored visualization of the normal map
    vis = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite('normal_map_test.png', vis)
    
    # Create a visualization of the mask
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite('mask_test.png', mask_vis)
    
    print("Test complete. Check normal_map_test.png and mask_test.png for visualizations.")

if __name__ == "__main__":
    test_normal_conversion() 