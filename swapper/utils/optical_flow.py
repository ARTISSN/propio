import cv2
import numpy as np
from pathlib import Path
import argparse

def compute_confidence_map(flow_forward, flow_backward):
    """
    Compute confidence map using forward-backward consistency check.
    
    Args:
        flow_forward: Forward optical flow (A->B)
        flow_backward: Backward optical flow (B->A)
        
    Returns:
        confidence_map: Higher values indicate more reliable flow estimates
    """
    h, w = flow_forward.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Warp backward flow using forward flow
    flow_backward_warped = cv2.remap(
        flow_backward,
        x + flow_forward[..., 0],
        y + flow_forward[..., 1],
        cv2.INTER_LINEAR
    )
    
    # Compute error as Euclidean distance between forward flow and warped backward flow
    error = np.sqrt(
        np.sum(
            (flow_forward + flow_backward_warped) ** 2,
            axis=-1
        )
    )
    
    # Convert error to confidence (higher error = lower confidence)
    confidence = 1.0 / (1.0 + error)
    
    return confidence

def compute_optical_flow_for_dir(
    img_dir,
    output_path,
    pattern="*.png"
):
    """
    Computes optical flow for all consecutive image pairs in a directory using TVL1 method.

    Args:
        img_dir (str or Path): Directory containing images.
        output_path (str or Path): Path to save the output .npz file.
        pattern (str): Glob pattern for images.
    """
    img_dir = Path(img_dir)
    output_path = Path(output_path)
    img_paths = sorted(img_dir.glob(pattern))
    if len(img_paths) < 2:
        raise ValueError("Need at least two images to compute optical flow.")

    flows = {}
    confidences = {}
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    
    for i in range(len(img_paths) - 1):
        img1 = cv2.imread(str(img_paths[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_paths[i+1]), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            print(f"Warning: Could not read {img_paths[i]} or {img_paths[i+1]}")
            continue

        # Forward flow (A->B)
        flow_forward = tvl1.calc(img1, img2, None)
        # Backward flow (B->A)
        flow_backward = tvl1.calc(img2, img1, None)
        
        # Compute confidence map
        confidence = compute_confidence_map(flow_forward, flow_backward)

        key = f"{img_paths[i].stem}_to_{img_paths[i+1].stem}"
        flows[key] = flow_forward.astype(np.float32)
        confidences[key] = confidence.astype(np.float32)

    # Save all flows and confidence maps in a single .npz file
    np.savez_compressed(output_path, flows=flows, confidences=confidences)
    print(f"Saved optical flow data and confidence maps to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute optical flow for a directory of images using TVL1 method.")
    parser.add_argument("img_dir", type=str, help="Directory containing images")
    parser.add_argument("output_path", type=str, help="Output .npz file for flow data")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for images")
    args = parser.parse_args()

    compute_optical_flow_for_dir(
        img_dir=args.img_dir,
        output_path=args.output_path,
        pattern=args.pattern
    )
