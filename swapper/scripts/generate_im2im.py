import os
import glob
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# -----------------------------------------------------------------------------
# Core Classes & Functions
# -----------------------------------------------------------------------------

class TemporalAdapter(torch.nn.Module):
    """
    A simple single-scale adapter that takes:
      - current warped features: List[Tensor] or single Tensor
      - previous smoothed features: same structure (or None for first frame)
    and outputs smoothed features of the same shape.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Example: a single linear layer per feature map channel
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, curr_feats, prev_feats):
        # curr_feats: Tensor[B, C, H, W] or list of such tensors
        # prev_feats: same shape (smoothed from last frame), or None
        if prev_feats is None:
            return curr_feats
        # simple residual smoothing:
        return curr_feats + self.linear(prev_feats.permute(0,2,3,1)).permute(0,3,1,2)

def load_image(path, size):
    """
    Load an image from disk, resize to `size=(w,h)`, convert to float tensor [0,1].
    Returns: Tensor[1,3,H,W], device-agnostic.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return tensor

def save_image(tensor, path):
    """
    Save a float Tensor[1,3,H,W] in [0,1] as a PNG.
    """
    arr = (tensor.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def compute_psnr(img_a, img_b):
    """
    Compute PSNR (dB) between two float Tensors [1,3,H,W].
    """
    mse = torch.mean((img_a - img_b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def warp_with_flow(feats, flow, device):
    """
    Warp feature maps by optical flow.

    Args:
      feats: Tensor[1,C,H,W]
      flow:  numpy array [H,W,2] with (dx,dy) per pixel
      device: torch device

    Returns:
      warped_feats: Tensor[1,C,H,W]
    """
    # TODO: implement via torch.nn.functional.grid_sample
    #   1. normalize flow to [-1,1] grid coords
    #   2. build grid of shape [1,H,W,2]
    #   3. grid_sample(feats, grid)
    raise NotImplementedError("warp_with_flow must be implemented")

def encode_and_extract(pipe, image, device):
    """
    Run img2img pass through the pipeline to get:
      - latents: Tensor latent codes before decoding
      - feats:   Tensor[1,C,H,W] or List[Tensor] of intermediate UNet feature maps

    This requires:
      1. Registering a forward hook on pipe.unet to capture hidden states.
      2. Calling pipe(image, **img2img_kwargs) to populate those hooks.
      3. Returning latents and the captured features.

    Returns:
      latents, feats
    """
    # TODO: 
    #   - set up `hidden_states = []`
    #   - define hook on each UNet block to append its output
    #   - call: pipe(
    #         prompt=None,
    #         init_image=image,
    #         strength=1.0,
    #         guidance_scale=cfg.guidance_scale,
    #         num_inference_steps=cfg.steps,
    #         output_type="latent",
    #     )
    #   - extract latents from pipe output
    #   - gather feats = hidden_states
    raise NotImplementedError("encode_and_extract must be implemented")

def decode_latents(pipe, latents):
    """
    Decode latents or feature maps back to RGB image tensor [1,3,H,W].
    If passing feature maps, you may need to call pipe.vae.decode directly.
    """
    # TODO:
    #   - If latents: return pipe.decode_latents(latents)
    #   - If feats: map feats back through UNet/VAE decode
    raise NotImplementedError("decode_latents must be implemented")


# -----------------------------------------------------------------------------
# Main Inference + Temporal Training Loop
# -----------------------------------------------------------------------------

def main(args):
    # —————————————————————————————
    # 1. Initialize pipeline & adapter
    # —————————————————————————————
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(args.spatial_lora)
    pipe.to(args.device).eval()
    # freeze spatial UNet
    for p in pipe.unet.parameters():
        p.requires_grad = False

    adapter = TemporalAdapter(hidden_dim=pipe.unet.config.hidden_size).to(args.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    # —————————————————————————————
    # 2. Gather input frames & flows
    # —————————————————————————————
    render_paths = sorted(glob.glob(os.path.join(args.renders_dir, "*.png")))
    flow_paths   = sorted(glob.glob(os.path.join(args.flow_dir,   "*.exr")))
    os.makedirs(args.output_dir, exist_ok=True)

    prev_feats = None

    # —————————————————————————————
    # 3. Frame Loop
    # —————————————————————————————
    for idx, (rpath, fpath) in enumerate(zip(render_paths, flow_paths)):
        # 3a. Load & preprocess render
        image = load_image(rpath, size=(args.resolution, args.resolution)).to(args.device)

        # 3b. Encode + extract features
        latents, feats = encode_and_extract(pipe, image, args.device)

        # 3c. Warp features using optical flow map
        flow = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        warped_feats = warp_with_flow(feats, flow, args.device)

        # 3d. Adapter forward + decode
        adapter.train()
        optimizer.zero_grad()
        smoothed = adapter(warped_feats, prev_feats)
        recon   = decode_latents(pipe, smoothed)

        # 3e. Conditional update based on PSNR
        with torch.no_grad():
            direct = decode_latents(pipe, feats)
        psnr = compute_psnr(recon, direct)
        if psnr < args.psnr_thresh:
            loss = torch.nn.functional.mse_loss(recon, direct)
            loss.backward()
            optimizer.step()

        # 3f. Save output frame
        out_path = os.path.join(args.output_dir, f"frame_{idx:04d}.png")
        save_image(recon, out_path)

        # 3g. Update for next iteration
        prev_feats = smoothed.detach()

    # —————————————————————————————
    # 4. Finalize
    # —————————————————————————————
    torch.save(adapter.state_dict(),
               os.path.join(args.output_dir, "temporal_adapter.ckpt"))
    print(f"Done! Frames and adapter saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=True,
                        help="SDXL model repo or path")
    parser.add_argument("--spatial-lora",  required=True,
                        help="Pretrained spatial LoRA checkpoint")
    parser.add_argument("--renders_dir",   required=True,
                        help="Directory of input render PNGs")
    parser.add_argument("--flow_dir",      required=True,
                        help="Directory of optical-flow EXR maps")
    parser.add_argument("--output_dir",    required=True,
                        help="Where to save output frames & adapter")
    parser.add_argument("--resolution",    type=int, default=512,
                        help="Square resolution for VAE & UNet")
    parser.add_argument("--psnr-thresh",   type=float, default=30.0,
                        help="PSNR threshold (dB) for adapter updates")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="Learning rate for adapter")
    parser.add_argument("--device",        default="cuda",
                        help="Torch device")
    args = parser.parse_args()
    main(args)