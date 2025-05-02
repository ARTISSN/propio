import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from scipy.special import sph_harm
from pathlib import Path
import datetime
from utils.spherical_gaussians import fit_spherical_gaussians

# Helper: compute SH basis up to given order
def compute_spherical_harmonics_basis(normals, order=3):
    """
    normals: (M,3) array of unit vectors
    returns: (M, (order+1)^2) basis matrix
    """
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi   = np.arctan2(y, x) % (2*np.pi)
    num_coeffs = (order + 1)**2
    basis = np.zeros((len(normals), num_coeffs), dtype=np.float32)
    idx = 0
    for l in range(order+1):
        for m in range(-l, l+1):
            basis[:, idx] = np.real(sph_harm(m, l, phi, theta))
            idx += 1
    return basis

# Render SH-lit normal map in color
def render_sh_lit_image(normal_map, coeffs, order=3):
    """
    normal_map: HxWx3 RGB in [-1,1]
    coeffs:    (C,3) array of SH coefficients for R,G,B (C=(order+1)^2)
    returns:   HxWx3 uint8 BGR image of lit normals
    """
    H, W, _ = normal_map.shape
    normals = normal_map.reshape(-1,3)
    basis = compute_spherical_harmonics_basis(normals, order)
    # compute color per channel
    lit_rgb = basis.dot(coeffs)                # (M,3)
    # clamp to [0,1]
    lit_rgb = np.clip(lit_rgb, 0.0, 1.0)
    lit_img = lit_rgb.reshape(H, W, 3)
    # convert RGB to BGR uint8
    lit_bgr = (lit_img[..., ::-1] * 255).astype(np.uint8)
    return lit_bgr

# Extract top-K sun directions and colors from SH coeffs
def sh_coeffs_to_suns(coeffs, order=3, K=5, sample_count=5000):
    """
    coeffs: (C,3) SH coefficients
    returns: list of dicts with 'direction', 'color', 'intensity'
    """
    # Fibonacci sampling on sphere
    i = np.arange(sample_count)
    phi   = np.arccos(1 - 2*(i+0.5)/sample_count)
    theta = 2 * np.pi * ((i+0.5)/((1+5**0.5)/2))
    theta %= 2*np.pi
    # build basis per sample
    lm = []
    for l in range(order+1):
        for m in range(-l, l+1):
            lm.append((l,m))
    B = np.stack([np.real(sph_harm(m, l, theta, phi)) for (l,m) in lm], axis=1)
    # predicted RGB
    pred_rgb = B.dot(coeffs)                    # (N,3)
    # compute luminance for ranking
    Yw = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lum = pred_rgb.dot(Yw)
    # select top K by luminance
    idx = np.argsort(lum)[-K:][::-1]
    suns = []
    for j in idx:
        th, ph = phi[j], theta[j]
        # direction in Cartesian
        x = np.sin(th)*np.cos(ph)
        y = np.sin(th)*np.sin(ph)
        z = np.cos(th)
        suns.append({
            'direction': (x, y, z),
            'color':     tuple(pred_rgb[j].tolist()),
            'intensity': float(lum[j])
        })
    return suns

# Main: calculate coefficients, save lit image, return per-channel coeffs
def calculate_lighting_coefficients(face_img, normal_map, order=3, save_path=None):
    """
    Fits 3-channel SH coefficients to normal_map + face_img.
    Optionally saves a preview of the normal map lit by the SH fit.

    Returns:
      coeffs: (C,3) array of R,G,B SH coefficients
    """
    # prepare normal_map in RGB [-1,1]
    nm_bgr = normal_map
    nm_rgb = cv2.cvtColor(nm_bgr, cv2.COLOR_BGR2RGB)
    nm = 2.0*(nm_rgb.astype(np.float32)/255.0) - 1.0

    # get linear RGB face colors
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # mask invalid normals
    mask = np.linalg.norm(nm, axis=2) > 0.1
    norms = nm[mask].reshape(-1,3)
    cols  = face_rgb[mask].reshape(-1,3)

    # subsample if too large
    M = len(norms)
    if M > 20000:
        idx = np.random.choice(M, 20000, replace=False)
        norms = norms[idx]
        cols  = cols[idx]

    # build SH basis
    basis = compute_spherical_harmonics_basis(norms, order)
    # fit multi-output linear regression without intercept
    model = LinearRegression(fit_intercept=False).fit(basis, cols)
    # model.coef_ shape is (3, C)
    coeffs = model.coef_.T                # shape (C,3)

    # save lit preview if requested
    if save_path is not None:
        lit_bgr = render_sh_lit_image(nm, coeffs, order)
        outp = Path(save_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), lit_bgr)

    return coeffs

class LightingProcessor:
    """Class to handle lighting coefficient calculation and management."""
    
    def __init__(self, base_path: Path, character_name: str):
        self.base_path = Path(base_path)
        self.character_name = character_name
        self.char_path = self.base_path / "characters" / character_name
    
    def process_frame(self, frame_id: str, maps_dir: Path):
        """Process a single frame and return lighting coefficients with metadata."""
        try:
            # Read the maps
            face_img_path = maps_dir / "faces" / (frame_id + ".png")
            normal_map_path = maps_dir / "normals" / (frame_id + ".png")
            face_img = cv2.imread(str(face_img_path))
            normal_map = cv2.imread(str(normal_map_path))
            
            if face_img is None or normal_map is None:
                raise ValueError(f"Could not read images for frame {frame_id}")
            
            # Calculate lighting coefficients
            #coeffs = calculate_lighting_coefficients(face_img, normal_map)
            # coefficients + save lit preview
            coeffs = calculate_lighting_coefficients(face_img, normal_map, order=3, save_path=str(maps_dir / "lighting" / (frame_id + ".png")))

            # convert SH to blender suns
            suns = sh_coeffs_to_suns(coeffs, order=3, K=5)
            
            # Create frame data
            frame_data = {
                "frame_id": frame_id,
                "suns": suns,
                "face_map": str(face_img_path.relative_to(self.base_path)),
                "normal_map": str(normal_map_path.relative_to(self.base_path)),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return frame_data
            
        except Exception as e:
            print(f"Error processing lighting for frame {frame_id}: {str(e)}")
            return None
