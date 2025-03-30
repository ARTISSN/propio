import numpy as np
from scipy.special import sph_harm

def estimate_lighting(normal_map, mask, num_bands=8):
    """
    Estimate lighting coefficients using spherical harmonics with hemisphere sampling.
    
    Args:
        normal_map: HxWx3 array of surface normals (in RGB format)
        mask: HxW boolean mask indicating valid pixels
        num_bands: Number of spherical harmonics bands to use (default: 3)
    
    Returns:
        lighting_coeffs: Array of spherical harmonics coefficients
    """
    # Convert normal map to normalized vectors
    normals = normal_map[mask] / 255.0  # Normalize to [0,1]
    normals = 2.0 * normals - 1.0  # Convert to [-1,1]
    
    # Normalize vectors
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    # Convert to spherical coordinates
    phi = np.arccos(normals[:, 2])  # Polar angle
    theta = np.arctan2(normals[:, 1], normals[:, 0])  # Azimuthal angle
    
    # Create hemisphere mask (only consider upward-facing normals)
    hemisphere_mask = normals[:, 2] > 0
    
    # Compute spherical harmonics basis functions
    Y = []
    for l in range(num_bands):
        for m in range(-l, l+1):
            if m < 0:
                Y.append(np.sqrt(2) * np.real(sph_harm(m, l, phi[hemisphere_mask], theta[hemisphere_mask])))
            elif m == 0:
                Y.append(np.real(sph_harm(m, l, phi[hemisphere_mask], theta[hemisphere_mask])))
            else:
                Y.append(np.sqrt(2) * np.imag(sph_harm(m, l, phi[hemisphere_mask], theta[hemisphere_mask])))
    
    Y = np.stack(Y, axis=1)
    
    # Create target lighting values (normalized dot product with assumed light direction)
    light_direction = np.array([0.0, 0.0, 1.0])  # Light from above
    target_lighting = np.dot(normals[hemisphere_mask], light_direction)
    
    # Normalize target lighting to [-1, 1] range
    target_lighting = 2.0 * (target_lighting - np.min(target_lighting)) / (np.max(target_lighting) - np.min(target_lighting)) - 1.0
    
    # Solve for lighting coefficients using regularized least squares
    # Add small regularization term to avoid overfitting
    lambda_reg = 0.01
    A = Y.T @ Y + lambda_reg * np.eye(Y.shape[1])
    b = Y.T @ target_lighting
    lighting_coeffs = np.linalg.solve(A, b)
    
    return lighting_coeffs

def calculate_lighting_from_normals(normal_map, lighting_coeffs, mask, num_bands = 5):
    """
    Calculate lighting values for each point in the normal map using spherical harmonic coefficients.
    
    Args:
        normal_map: HxWx3 array of surface normals (in RGB format)
        lighting_coeffs: Array of spherical harmonic coefficients
        mask: HxW boolean mask indicating valid pixels
        
    Returns:
        lighting_map: HxW array of lighting values
    """
    # Convert normal map to normalized vectors
    normals = normal_map / 255.0  # Normalize to [0,1]
    normals = 2.0 * normals - 1.0  # Convert to [-1,1]
    
    # Extract valid normals using the mask
    valid_normals = normals[mask]
    
    # Normalize vectors
    norms = np.linalg.norm(valid_normals, axis=1, keepdims=True)
    valid_normals = valid_normals / (norms + 1e-8)
    
    # Convert to spherical coordinates
    phi = np.arccos(valid_normals[:, 2])  # Polar angle (0 to π)
    theta = np.arctan2(valid_normals[:, 1], valid_normals[:, 0])  # Azimuthal angle (0 to 2π)
    
    # Initialize lighting values
    lighting = np.zeros(mask.shape)
    valid_lighting = np.zeros(len(valid_normals))
    
    # Compute lighting using spherical harmonics
    for l in range(num_bands):  # Use 3 bands
        for m in range(-l, l+1):
            idx = l*(l+1) + m
            if idx >= len(lighting_coeffs):
                continue
                
            # Calculate spherical harmonic term for valid pixels only
            if m < 0:
                term = lighting_coeffs[idx] * np.sqrt(2) * np.real(sph_harm(m, l, theta, phi))
            elif m == 0:
                term = lighting_coeffs[idx] * np.real(sph_harm(m, l, theta, phi))
            else:
                term = lighting_coeffs[idx] * np.sqrt(2) * np.imag(sph_harm(m, l, theta, phi))
            
            # Add term to valid lighting (both should be shape (48771,))
            valid_lighting += term

    # Negate before normalization
    valid_lighting = -valid_lighting
    
    # Normalize valid lighting to [0,1] range
    if np.max(valid_lighting) > np.min(valid_lighting):
        valid_lighting = (valid_lighting - np.min(valid_lighting)) / (np.max(valid_lighting) - np.min(valid_lighting))
    
    # Add ambient lighting
    ambient_light = 0.2  # Ambient lighting coefficient
    valid_lighting = ambient_light + (1 - ambient_light) * valid_lighting
    
    # Place valid lighting values back into full lighting map
    lighting[mask] = valid_lighting
    
    # Convert to uint8 format
    lighting_map = (lighting * 255).astype(np.uint8)
    
    return lighting_map 