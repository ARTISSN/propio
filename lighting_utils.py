import numpy as np
from scipy.special import sph_harm
import cv2
import time
import coordinate_utils
from coordinate_utils import rgb_to_xyz, xyz_to_spherical, normalize_vector, xyz_to_rgb
import matplotlib.pyplot as plt

def compute_spherical_harmonics(theta, phi, num_bands, debug=False):
    """
    Compute spherical harmonics basis functions for given spherical coordinates.
    
    Args:
        theta: azimuthal angles
        phi: polar angles
        num_bands: number of spherical harmonic bands (l=0 to num_bands-1)
        debug: whether to print debug information
        
    Returns:
        Y: matrix of shape (N, num_coeffs) containing spherical harmonic values
    """
    Y = []
    for l in range(num_bands):
        for m in range(-l, l+1):
            sh = sph_harm(m, l, phi, theta)
            if m < 0:
                term = np.sqrt(2) * np.real(sh)
            elif m == 0:
                term = np.real(sh)
            else:
                term = np.sqrt(2) * np.imag(sh)
            # Ensure term is 1D array
            Y.append(term.reshape(-1))
    
    # Stack terms as columns
    Y = np.stack(Y, axis=1)
    
    if debug:
        print(f"\nSpherical Harmonics computed:")
        print(f"Shape: {Y.shape}")
        print("Sample values:")
        for i in range(min(3, Y.shape[0])):
            print(f"Row {i}: {Y[i]}")
    
    return Y

def estimate_lighting(normal_map, mask, num_bands=3, debug=False):
    """
    Estimate lighting from a normal map using spherical harmonics.
    
    Args:
        normal_map: Normal map in RGB format
        mask: Binary mask indicating valid pixels
        num_bands: Number of spherical harmonics bands to use (default: 3)
        debug: If True, print debug information
        
    Returns:
        tuple: (lighting_coefficients, lighting_map, Y)
            lighting_coefficients: Array of spherical harmonic coefficients
            lighting_map: HxW array of lighting values
            Y: Spherical harmonics basis functions
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
    
    # Convert benchmark normals to RGB using our coordinate system
    benchmark_rgb = np.zeros_like(benchmark_xyz)
    benchmark_rgb[:, 0] = (benchmark_xyz[:, 0] + 1) * 127.5  # X -> R
    benchmark_rgb[:, 1] = (benchmark_xyz[:, 1] + 1) * 127.5  # Y -> G
    benchmark_rgb[:, 2] = (1 - benchmark_xyz[:, 2]) * 127.5  # Z -> B
    
    if debug:
        print("\n=== Benchmark Normal Values ===")
        print("Direction | XYZ Coordinates | RGB Values")
        print("----------|----------------|------------")
        directions = ["Forward", "Back", "Left", "Right", "Up", "Down"]
        for i, (xyz, rgb) in enumerate(zip(benchmark_xyz, benchmark_rgb)):
            print(f"{directions[i]:<9} | {xyz} | {rgb.astype(int)}")
    
    # 1. Convert normal map to XYZ
    xyz_normals = coordinate_utils.rgb_to_xyz(normal_map[mask])
    if debug:
        print("\n1. Converted to XYZ coordinates")
        print(f"Shape: {xyz_normals.shape}")
        print("Sample values:")
        for i in range(min(5, len(xyz_normals))):
            print(f"   {xyz_normals[i]}")
    
    # 2. Normalize the vectors
    normalized_xyz = coordinate_utils.normalize_vector(xyz_normals)
    if debug:
        print("\n2. Normalized vectors")
        print("Sample values:")
        for i in range(min(5, len(normalized_xyz))):
            print(f"   {normalized_xyz[i]}")
    
    # 3. Convert to spherical coordinates
    theta, phi = coordinate_utils.xyz_to_spherical(normalized_xyz)
    if debug:
        print("\n3. Converted to spherical coordinates")
        print("Sample values:")
        for i in range(min(5, len(theta))):
            print(f"   θ={theta[i]:.2f}, φ={phi[i]:.2f}")
    
    # 4. Compute spherical harmonics
    Y = compute_spherical_harmonics(theta, phi, num_bands, debug)
    
    # 5. Create target lighting values (white light)
    target_lighting = np.ones(len(normalized_xyz))
    if debug:
        print("\n5. Created target lighting values")
        print("Sample values:")
        for i in range(min(5, len(target_lighting))):
            print(f"   {target_lighting[i]}")
    
    # 6. Solve for lighting coefficients
    lambda_reg = 0.01
    A = Y.T @ Y + lambda_reg * np.eye(Y.shape[1])
    b = Y.T @ target_lighting
    lighting_coeffs = np.linalg.solve(A, b)
    if debug:
        print("\n6. Solved for lighting coefficients")
        print(f"Shape: {lighting_coeffs.shape}")
        print(f"Values: {lighting_coeffs}")
    
    # 7. Create lighting map
    scalar_lighting = np.clip(
        np.dot(Y, lighting_coeffs),
        0,
        1
    )
    lighting_map = np.zeros_like(normal_map)
    lighting_map[mask] = np.repeat(scalar_lighting[:, np.newaxis], 3, axis=1)
    
    if debug:
        print("\n7. Created lighting map")
        print(f"Shape: {lighting_map.shape}")
        print("Sample values:")
        for i in range(min(5, len(scalar_lighting))):
            print(f"   {scalar_lighting[i]:.3f}")
    
    return lighting_coeffs, lighting_map, Y

def calculate_lighting_from_normals(normal_map, lighting_coeffs, mask, num_bands=None, debug=False):
    """
    Calculate lighting values for each point in the normal map using pre-computed spherical harmonic coefficients.
    This function is typically used after estimate_lighting() to apply the same lighting to different normal maps.
    
    Args:
        normal_map: HxWx3 array of surface normals (in RGB format)
        lighting_coeffs: Array of spherical harmonic coefficients (from estimate_lighting())
        mask: HxW boolean mask indicating valid pixels
        num_bands: Number of spherical harmonics bands to use. If None, inferred from lighting_coeffs
        debug: If True, print debug information
        
    Returns:
        lighting_map: HxW array of lighting values
    """
    # Infer number of bands from lighting coefficients if not provided
    if num_bands is None:
        num_coeffs = len(lighting_coeffs)
        # For l=0,1: 4 coeffs, l=0,1,2: 9 coeffs, l=0,1,2,3: 16 coeffs
        num_bands = int((np.sqrt(1 + 4*num_coeffs) - 1) / 2)
    
    if debug:
        print(f"\nInferred {num_bands} bands from {len(lighting_coeffs)} coefficients")
    
    # Convert normal map to XYZ
    xyz_normals = coordinate_utils.rgb_to_xyz(normal_map[mask])
    
    # Normalize the vectors
    normalized_xyz = coordinate_utils.normalize_vector(xyz_normals)
    
    # Convert to spherical coordinates
    theta, phi = coordinate_utils.xyz_to_spherical(normalized_xyz)
    
    # Compute spherical harmonics using shared function
    Y = compute_spherical_harmonics(theta, phi, num_bands, debug)
    
    # Calculate lighting values
    scalar_lighting = np.clip(
        np.dot(Y, lighting_coeffs),
        0,
        1
    )
    
    # Create final lighting map
    lighting_map = np.zeros_like(normal_map)
    lighting_map[mask] = np.repeat(scalar_lighting[:, np.newaxis], 3, axis=1)
    
    return lighting_map

def visualize_single_sphere(lighting_coeffs, size=150, debug=False):
    """
    Visualize a single sphere with given lighting coefficients.
    Returns the sphere image.
    """
    # Create sphere
    normal_map, mask = create_spherical_normal_map(size)
    
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)
    if debug:
        cv2.imwrite("sphere_norm.png", normal_map)
        
    lighting_bgr = calculate_lighting_from_normals(normal_map, lighting_coeffs, mask, debug=debug)
    
    return lighting_bgr

def create_spherical_normal_map(size=100):
    """
    Create a normal map representing a sphere.
    Returns normal map and mask.
    """
    normal_map = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    
    # Create a grid of coordinates
    y, x = np.mgrid[0:size, 0:size]
    x = (x - size/2) / (size/2)  # Normalize to [-1, 1]
    y = (y - size/2) / (size/2)  # Normalize to [-1, 1]
    
    # Calculate radius from center
    r = np.sqrt(x*x + y*y)
    
    # Create mask for points within unit circle
    mask = r < 1.0  # Changed from <= to < to avoid edge cases
    
    # For points inside the sphere, calculate z coordinate
    z = np.zeros_like(r)
    # Only calculate z for points where r < 1
    valid_points = r < 1.0  # Changed from <= to < to avoid edge cases
    z[valid_points] = -np.sqrt(1 - x[valid_points]**2 - y[valid_points]**2)  # Negative for forward-facing
    
    # Convert to XYZ coordinates
    xyz = np.stack([x[valid_points], y[valid_points], z[valid_points]], axis=-1)
    xyz[:,2] = -xyz[:,2]
    xyz = normalize_vector(-xyz)
    
    # Convert XYZ to RGB for storage
    rgb = xyz_to_rgb(xyz)
    
    # Store in normal map
    normal_map[valid_points] = rgb.astype(np.uint8)
    
    return normal_map, mask

def visualize_benchmark_lighting_on_normal_map(normal_map, mask):
    """
    Visualize lighting from 6 different directions on the input normal map.
    Returns the combined image.
    """
    height, width = normal_map.shape[:2]
    padding = 20  # Padding between visualizations
    
    # Create a combined image with 2x3 grid
    combined_height = 2 * height + 3 * padding
    combined_width = 3 * width + 4 * padding
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Define lighting coefficients for different directions
    directions = [
        ("Up (positive Y)", [0.5, 0.0, 1.0, 0.0]),  # l=1, m=0
        ("Down (negative Y)", [0.5, 0.0, -1.0, 0.0]),
        ("Right (positive X)", [0.5, 0.0, 0.0, 1.0]),  # l=1, m=1
        ("Left (negative X)", [0.5, 0.0, 0.0, -1.0]),
        ("Forward (positive Z)", [0.5, 1.0, 0.0, 0.0]),  # l=1, m=-1
        ("Back (negative Z)", [0.5, -1.0, 0.0, 0.0])
    ]
    
    # Generate and place each lighting visualization
    for idx, (direction_name, coeffs) in enumerate(directions):
        # Calculate lighting for this direction
        lighting_bgr = calculate_lighting_from_normals(normal_map, coeffs, mask)
        
        # Convert to BGR for OpenCV
        #lighting_bgr = cv2.cvtColor(lighting_map, cv2.COLOR_GRAY2BGR)
        
        # Calculate position in grid
        row = idx // 3
        col = idx % 3
        
        # Place lighting visualization in combined image
        y_start = row * (height + padding) + padding
        x_start = col * (width + padding) + padding
        combined_image[y_start:y_start+height, x_start:x_start+width] = lighting_bgr
        
        # Add text label
        cv2.putText(combined_image, direction_name, 
                   (x_start, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return combined_image