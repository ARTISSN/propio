import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import least_squares

# Helper functions
def load_normal_map(normal_map):
    nm = normal_map.astype(np.float32) / 255.0
    normals = nm[..., :3] * 2.0 - 1.0
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(lengths, 1e-6, None)


def load_and_linearize_image(image):
    srgb = image.astype(np.float32) / 255.0
    mask = (srgb <= 0.04045)
    lin = np.empty_like(srgb)
    lin[mask]  = srgb[mask]  / 12.92
    lin[~mask] = ((srgb[~mask] + 0.055) / 1.055) ** 2.4
    return lin


def equal_area_binning(normals, colors, B):
    """
    Bin normals/colors into ~B equal-area cells on the sphere.
    Returns bin normals, bin colors, and counts per bin.
    """
    # Flatten inputs
    n = normals.reshape(-1, 3)
    c = colors.reshape(-1, 3)
    M = n.shape[0]
    # Determine lat/lon resolution
    n_lat = int(np.sqrt(B/2))
    n_lon = int(np.ceil(B / n_lat))
    # Spherical coords
    cos_th = n[:, 2]
    phi    = np.arctan2(n[:, 1], n[:, 0]) % (2*np.pi)
    # Bin edges
    lat_edges = np.linspace(-1.0, 1.0, n_lat+1)
    lon_edges = np.linspace(0.0, 2*np.pi, n_lon+1)
    # Assign bins
    lat_idx = np.clip(np.digitize(cos_th, lat_edges)-1, 0, n_lat-1)
    lon_idx = np.clip(np.digitize(phi,    lon_edges)-1, 0, n_lon-1)
    bin_idx = lat_idx * n_lon + lon_idx
    B_tot   = n_lat * n_lon
    # Accumulate
    bin_n = np.zeros((B_tot, 3), dtype=np.float32)
    bin_c = np.zeros((B_tot, 3), dtype=np.float32)
    cnt   = np.zeros(B_tot,    dtype=np.int32)
    for i, b in enumerate(bin_idx):
        bin_n[b] += n[i]
        bin_c[b] += c[i]
        cnt[b]  += 1
    # Keep non-empty
    mask = cnt > 0
    bin_n = bin_n[mask]
    bin_c = bin_c[mask] / cnt[mask, None]
    cnt   = cnt[mask].astype(np.float32)
    # Renormalize normals
    norms = np.linalg.norm(bin_n, axis=1, keepdims=True)
    bin_n /= np.clip(norms, 1e-6, None)
    return bin_n, bin_c, cnt


def compute_brightness_scale(mus, lambs, weights, bin_n, bin_c, counts):
    """
    Compute scale factor so that mean luminance of SG fit matches observed.
    """
    # Luminance weights
    Yw = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # Observed mean luminance
    L_obs = (counts * (bin_c @ Yw)).sum() / counts.sum()
    # Predicted per-bin color
    D = bin_n @ np.stack(mus).T                # (B, K)
    G = np.exp(lambs * (D - 1.0))              # (B, K)
    pred_c = G @ np.stack(weights)            # (B, 3)
    L_pred = (counts * (pred_c @ Yw)).sum() / counts.sum()
    # Return scale
    return L_obs / (L_pred + 1e-8)


def fit_spherical_gaussians(normal_map: np.ndarray,
                            reference_image: np.ndarray,
                            K: int = 4,
                            subsample: int = 20_000):
    """
    Fit a mixture of K spherical Gaussians to normals+image and
    apply a brightness scale so fit matches observed luminance.

    Returns:
      mus:    (K,3) unit directions
      lambs:  (K,) sharpness
      weights:(K,3) scaled RGB weights
    """
    # Load and preprocess
    normals = load_normal_map(normal_map)
    colors  = load_and_linearize_image(reference_image)
    # Flatten all pixels
    n = normals.reshape(-1, 3)
    c = colors.reshape(-1, 3)
    # Equal-area binning or full set
    if n.shape[0] > subsample:
        bin_n, bin_c, counts = equal_area_binning(normals, colors, subsample)
        w_sqrt = np.sqrt(counts)[:, None]
    else:
        bin_n, bin_c, counts = n, c, np.ones(n.shape[0], dtype=np.float32)
        w_sqrt = None
    # Initialize mu via KMeans on bin normals
    brightness = bin_c.mean(axis=1)
    km = KMeans(n_clusters=K, random_state=0).fit(bin_n, sample_weight=brightness+1e-3)
    mus = km.cluster_centers_
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    # Initialize lambs and weights
    lambs   = np.full(K, 20.0, dtype=np.float32)
    weights = np.array([bin_c[km.labels_ == i].mean(axis=0) for i in range(K)], dtype=np.float32)
    # Pack params
    thetas = np.arccos(np.clip(mus[:, 2], -1, 1))
    phis   = np.arctan2(mus[:, 1], mus[:, 0]) % (2*np.pi)
    logls  = np.log(lambs)
    p0 = np.hstack([thetas, phis, logls, weights.ravel()])
    # Unpack helper
    def unpack(p):
        p = p.reshape(K, 6)
        θ, φ, logλ, wR, wG, wB = p.T
        λ = np.exp(logλ)
        μ = np.stack([
            np.sin(θ)*np.cos(φ),
            np.sin(θ)*np.sin(φ),
            np.cos(θ)
        ], axis=1)
        W = np.stack([wR, wG, wB], axis=1)
        return μ, λ, W
    # Residuals with bin weighting
    def residuals(p):
        μ, λ, W = unpack(p)
        D = bin_n @ μ.T
        G = np.exp(λ * (D - 1.0))
        pred = G @ W
        res  = pred - bin_c
        return (w_sqrt * res).ravel() if w_sqrt is not None else res.ravel()
    # Solve
    res = least_squares(residuals, p0, verbose=2, ftol=1e-6, xtol=1e-6, max_nfev=200)
    # Unpack final
    mus_fit, lambs_fit, weights_fit = unpack(res.x)
    # Apply brightness scale
    scale = compute_brightness_scale(mus_fit, lambs_fit, weights_fit, bin_n, bin_c, counts)
    print("scale", scale)
    weights_fit *= scale
    return mus_fit, lambs_fit, weights_fit

# Example usage remains unchanged
if __name__ == "__main__":
    import imageio
    normal_map = imageio.imread("normal_map.png")
    reference_image = imageio.imread("reference.png")
    mus, lambs, weights = fit_spherical_gaussians(
        normal_map, reference_image, K=5
    )
    for i in range(len(mus)):
        print(f"Lobe {i+1}: μ={mus[i]}, λ={lambs[i]:.2f}, w={weights[i]}")
