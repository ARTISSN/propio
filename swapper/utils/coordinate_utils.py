import numpy as np

def rgb_to_xyz(normals):
    """
    Convert normals from RGB format to XYZ coordinate system.
    Convention: 
    - X: -1 to +1 : Red: 0 to 255 : left to right
    - Y: -1 to +1 : Green: 0 to 255 : down to up
    - Z: 0 to -1 : Blue: 128 to 255 : neutral to forward (towards camera)
    
    Args:
        normals: Nx3 array of normalized vectors in RGB format
        
    Returns:
        Nx3 array of normalized vectors in XYZ format
    """

    remapped_normals = 2.0 * (normals / 255.0 - 0.5)
    # Map RGB to XYZ with proper scaling
    remapped_normals[:, 2] = -remapped_normals[:, 2]  # B -> Z (neutral to forward) 
    return remapped_normals

def xyz_to_rgb(normals):
    """
    Convert normals from XYZ coordinate system to RGB format.
    Convention:
    - X: -1 to +1 : Red: 0 to 255 : left to right
    - Y: -1 to +1 : Green: 0 to 255 : down to up
    - Z: 0 to -1 : Blue: 128 to 255 : neutral to forward (towards camera)
    
    Args:
        normals: Nx3 array of normalized vectors in XYZ format
        
    Returns:
        Nx3 array of normalized vectors in RGB format
    """
    remapped_normals = np.zeros_like(normals)
    # Map XYZ to RGB with proper scaling
    remapped_normals[:, 0] = (normals[:, 0] + 1.0) * 127.5  # X -> R (left to right)
    remapped_normals[:, 1] = (normals[:, 1] + 1.0) * 127.5  # Y -> G (down to up)
    remapped_normals[:, 2] = (1.0 - normals[:, 2]) * 127.5  # Z -> B (neutral to forward)
    return remapped_normals

def xyz_to_spherical(xyz):
    """
    Convert XYZ coordinates to spherical coordinates (theta, phi).
    Convention:
    - X: -1 to +1 : left to right
    - Y: -1 to +1 : down to up
    - Z: 0 to -1 : neutral to forward (towards camera)
    
    Args:
        xyz: Nx3 array of normalized vectors in XYZ format
        
    Returns:
        tuple: (theta, phi) where:
            - theta: azimuthal angle in XZ plane (0 to 2π)
            - phi: polar angle from Y axis (0 to π)
    """
    phi = np.arccos(xyz[:, 1])  # Polar angle (0 to π) from Y axis
    theta = np.arctan2(xyz[:, 0], -xyz[:, 2])  # Azimuthal angle (0 to 2π) in XZ plane
    return theta, phi

def spherical_to_xyz(theta, phi):
    """
    Convert spherical coordinates to XYZ coordinates.
    Convention:
    - X: -1 to +1 : left to right
    - Y: -1 to +1 : down to up
    - Z: 0 to -1 : neutral to forward (towards camera)
    
    Args:
        theta: azimuthal angle in XZ plane (0 to 2π)
        phi: polar angle from Y axis (0 to π)
        
    Returns:
        Nx3 array of normalized vectors in XYZ format
    """
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = -np.sin(phi) * np.cos(theta)  # Negative to make Z positive towards camera
    return np.stack([x, y, z], axis=1)

def normalize_vector(v):
    """
    Normalize a vector or array of vectors.
    
    Args:
        v: Nx3 array of vectors
        
    Returns:
        Nx3 array of normalized vectors
    """
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-8)

def debug_coordinate_conversions(normal_map=None, mask=None):
    """
    Comprehensive debug function for coordinate conversions and normal map processing.
    
    Args:
        normal_map: Optional normal map to test actual pipeline
        mask: Optional mask for normal map
    """
    # Define benchmark normals in XYZ format
    benchmark_xyz = np.array([
        [0, 0, -1],  # Forward
        [0, 0, 1],   # Back
        [-1, 0, 0],  # Left
        [1, 0, 0],   # Right
        [0, 1, 0],   # Up
        [0, -1, 0]   # Down
    ])
    
    print("\n=== Coordinate Conversion Debug ===")
    print("Testing XYZ -> RGB -> XYZ conversion:")
    print("Direction | Original XYZ | RGB | Reconstructed XYZ")
    print("----------|-------------|-----|-----------------")
    
    for i, xyz in enumerate(benchmark_xyz):
        # Convert to RGB
        rgb = xyz_to_rgb(xyz.reshape(1, 3))
        # Convert back to XYZ
        reconstructed_xyz = rgb_to_xyz(rgb)
        
        print(f"{['Forward', 'Back', 'Left', 'Right', 'Up', 'Down'][i]:<9} | {xyz} | {rgb[0].astype(int)} | {reconstructed_xyz[0]}")
    
    print("\nTesting XYZ -> Spherical -> XYZ conversion:")
    print("Direction | Original XYZ | Spherical | Reconstructed XYZ")
    print("----------|-------------|-----------|-----------------")
    
    for i, xyz in enumerate(benchmark_xyz):
        # Convert to spherical
        theta, phi = xyz_to_spherical(xyz.reshape(1, 3))
        # Convert back to XYZ
        reconstructed_xyz = spherical_to_xyz(theta, phi)
        
        print(f"{['Forward', 'Back', 'Left', 'Right', 'Up', 'Down'][i]:<9} | {xyz} | θ={theta[0]:.2f}, φ={phi[0]:.2f} | {reconstructed_xyz[0]}")
    
    # Test normalization
    print("\nTesting vector normalization:")
    print("Direction | Original XYZ | Normalized XYZ")
    print("----------|-------------|---------------")
    
    for i, xyz in enumerate(benchmark_xyz):
        normalized = normalize_vector(xyz.reshape(1, 3))
        print(f"{['Forward', 'Back', 'Left', 'Right', 'Up', 'Down'][i]:<9} | {xyz} | {normalized[0]}")
    
    # If normal map is provided, test the full pipeline
    if normal_map is not None and mask is not None:
        print("\n=== Normal Map Pipeline Debug ===")
        
        # Convert normal map to XYZ
        xyz_normals = rgb_to_xyz(normal_map[mask])
        
        # Normalize the vectors
        normalized_xyz = normalize_vector(xyz_normals)
        
        # Convert to spherical coordinates
        theta, phi = xyz_to_spherical(normalized_xyz)
        
        # Create hemisphere mask
        hemisphere_mask = theta < np.pi/2
        
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of valid pixels: {np.sum(mask)}")
        print(f"Number of upward-facing normals: {np.sum(hemisphere_mask)}")
        
        # Print some sample values
        print("\nSample normal map values:")
        print("Pixel | RGB | XYZ | Spherical")
        print("------|-----|-----|----------")
        for i in range(min(5, len(xyz_normals))):
            print(f"{i} | {normal_map[mask][i]} | {xyz_normals[i]} | θ={theta[i]:.2f}, φ={phi[i]:.2f}") 