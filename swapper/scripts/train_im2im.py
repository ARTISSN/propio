#!/usr/bin/env python3
"""
train_mesh2face_lora.py

Fine-tunes a Low-Rank Adapter (LoRA) on top of SDXL's U-Net
to learn a mesh→photoreal face mapping for your character.
Uses paired mesh renders (input) and FLUX-generated portraits (target).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
from pathlib import Path
import modal
import torch
if not hasattr(torch, "uint1"):
    torch.uint1 = torch.bool
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.nn import functional as F
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler, EulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torchvision import transforms as T
from PIL import Image
import yaml
from accelerate import Accelerator
import datetime
from diffusers.training_utils import EMAModel
from huggingface_hub import snapshot_download
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import get_cosine_schedule_with_warmup
import lpips
import torchvision
import numpy as np
import itertools
import gc
from contextlib import nullcontext

# Import checkpoint utilities
from utils.checkpoint_utils import save_lora_state, find_lora_checkpoint, load_checkpoint

# Global debug mode flag - will be set from config
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Wrapper for print that only outputs if DEBUG_MODE is True.
    
    This function behaves exactly like the built-in print() but only outputs
    if the global DEBUG_MODE flag is True.
    """
    if DEBUG_MODE:
        print(*args, **kwargs)

# ─── 1) CONFIGURATION ─────────────────────────────────────────────────────────

MOUNT_ROOT = "/workspace"
DATA_MOUNT = f"{MOUNT_ROOT}/data/characters"
CACHE_MOUNT = f"{MOUNT_ROOT}/cache"
CHARACTER_DATA_VOLUME = modal.Volume.from_name("character-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)  # Single volume for all caches

# ─── 2) CLASSES ────────────────────────────────────────────────────────────────
class GradStableDiffusionXLImg2ImgPipeline(StableDiffusionXLImg2ImgPipeline):
    def __call__(self, *args, **kwargs):
        grad_enabled = kwargs.get("gradient", False)
        # Call the original function, not the no_grad wrapper
        raw_call = super().__call__.__wrapped__
        if grad_enabled:
            with torch.enable_grad():
                return raw_call(self, *args, **kwargs)
        else:
            with torch.no_grad():
                return raw_call(self, *args, **kwargs)

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial loss."""
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for feat in features:
            layers.append(nn.Conv2d(prev_ch, feat, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_ch = feat
        layers.append(nn.Conv2d(prev_ch, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())       # <<< now D(x) ∈ [0,1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class TemporalDataset(Dataset):
    """Dataset for temporal consistency training using optical flow."""
    def __init__(self, mesh_dir, flow_npz_path, tf):
        self.mesh_paths = sorted(mesh_dir.glob("*.png"))
        try:
            self.flow_data = np.load(flow_npz_path)
        except Exception as e:
            print(f"Error loading flow data: {e}")
            self.flow_data = {}
        # keys like "frame0001_to_frame0002"
        self.flow_keys = sorted(self.flow_data.files)
        self.tf = tf

    def __len__(self):
        return len(self.flow_keys)

    def __getitem__(self, idx):
        # Parse the key to get the frame indices
        key = self.flow_keys[idx]
        # Example key: "frame0001_to_frame0002"
        frame1, frame2 = key.split("_to_")
        # Find the corresponding image paths
        img_t = next(p for p in self.mesh_paths if p.stem == frame1)
        img_t1 = next(p for p in self.mesh_paths if p.stem == frame2)
        flow = self.flow_data[key]  # H×W×2 float32
        return self.tf(Image.open(img_t).convert("RGB")), \
               self.tf(Image.open(img_t1).convert("RGB")), \
               torch.from_numpy(flow).permute(2,0,1)
    
class MeshFaceDataset(Dataset):
    """Dataset for mesh-to-face paired training."""
    def __init__(self, character_path: Path, transform, resolution=None, vae=None):
        # Load blacklist if it exists
        blacklist_path = character_path / "blacklist.txt"
        if blacklist_path.exists():
            with open(blacklist_path, "r") as f:
                blacklist = set(line.strip() for line in f if line.strip())
        else:
            blacklist = set()

        # Gather all renders and faces, filter by blacklist
        all_renders = sorted((character_path / "processed" / "maps" / "lighting").glob("*.png"))
        all_faces = sorted((character_path / "processed" / "faces").glob("*.png"))
        print(f"Found {len(all_renders)} renders and {len(all_faces)} faces")

        # Create dictionaries for easy lookup
        render_dict = {r.stem: r for r in all_renders}
        face_dict = {f.stem: f for f in all_faces}

        # Verify matching pairs
        self.renders = []
        self.faces = []
        mismatched_pairs = []
        
        for render_path in all_renders:
            stem = render_path.stem
            if stem in blacklist:
                continue
                
            if stem not in face_dict:
                mismatched_pairs.append(f"Render {stem} has no matching face")
                continue
                
            self.renders.append(render_path)
            self.faces.append(face_dict[stem])
            
        # Report any mismatches
        if mismatched_pairs:
            print("\n⚠️ Found mismatched pairs:")
            for msg in mismatched_pairs:
                print(f"  - {msg}")
            print(f"\nTotal mismatches: {len(mismatched_pairs)}")
            
        # Verify we have matching pairs
        assert len(self.renders) == len(self.faces), \
            f"Render and face counts do not match after applying blacklist! Renders: {len(self.renders)}, Faces: {len(self.faces)}"
            
        # Verify all pairs have matching stems
        for r, f in zip(self.renders, self.faces):
            assert r.stem == f.stem, f"Mismatched pair found: {r.stem} != {f.stem}"
            
        print(f"Successfully paired {len(self.renders)} matching render/face pairs")
        
        self.tf = transform
        self.character_path = character_path
        self.vae = vae
        self.update_resolution(resolution)

    def update_resolution(self, resolution=None):
        """Update latent directories based on resolution"""
        res_suffix = f"_{resolution}" if resolution else ""
        self.mesh_latent_dir = self.character_path / "processed" / "latents" / f"mesh{res_suffix}"
        self.ref_latent_dir  = self.character_path / "processed" / "latents" / f"ref{res_suffix}"
        
        # Use the stems from the lighting directory as frame IDs
        self.frame_ids = [p.stem for p in self.renders]
        
        # Check if resolution-specific latents exist, otherwise use default
        if not self.mesh_latent_dir.exists() or not self.ref_latent_dir.exists():
            debug_print(f"Warning: No latents found for resolution {resolution}, using default")
            self.mesh_latent_dir = self.character_path / "processed" / "latents" / "mesh"
            self.ref_latent_dir  = self.character_path / "processed" / "latents" / "ref"

    def __len__(self):
        return len(self.renders)

    def __getitem__(self, idx):
        render_img = Image.open(self.renders[idx]).convert("RGB")
        face_img  = Image.open(self.faces[idx]).convert("RGB")
        frame_id = self.frame_ids[idx]
        
        # Try to load latents if they exist
        try:
            mesh_latent = torch.load(self.mesh_latent_dir / f"{frame_id}.pt").to(torch.bfloat16)
            ref_latent  = torch.load(self.ref_latent_dir  / f"{frame_id}.pt").to(torch.bfloat16)
            
            # Handle different latent dimensions
            # Case 1: [C, H, W] -> add batch dim -> [1, C, H, W]
            if mesh_latent.dim() == 3:
                mesh_latent = mesh_latent.unsqueeze(0)
            if ref_latent.dim() == 3:
                ref_latent = ref_latent.unsqueeze(0)
            
            # Case 2: [B, 1, C, H, W] -> squeeze middle dim -> [B, C, H, W]
            if mesh_latent.dim() == 5:
                mesh_latent = mesh_latent.squeeze(1)
            if ref_latent.dim() == 5:
                ref_latent = ref_latent.squeeze(1)
            
            # Ensure we have exactly 4 dimensions [B, C, H, W]
            if mesh_latent.dim() != 4:
                raise ValueError(f"Mesh latent has wrong number of dimensions: {mesh_latent.dim()}, shape: {mesh_latent.shape}")
            if ref_latent.dim() != 4:
                raise ValueError(f"Ref latent has wrong number of dimensions: {ref_latent.dim()}, shape: {ref_latent.shape}")
            
        except FileNotFoundError:
            if self.vae is None:
                raise RuntimeError("VAE not provided for on-demand latent generation")
            # If latents don't exist, create them
            with torch.no_grad():
                mesh_latent = self.vae.encode(self.tf(render_img).unsqueeze(0)).latent_dist.sample() * self.vae.config.scaling_factor
                ref_latent = self.vae.encode(self.tf(face_img).unsqueeze(0)).latent_dist.sample() * self.vae.config.scaling_factor
                
                # Debug prints for generated latents
                debug_print(f"Generated latent shapes - mesh: {mesh_latent.shape}, ref: {ref_latent.shape}")
                
                # Ensure latents are 4D [B, C, H, W]
                if mesh_latent.dim() == 5:
                    mesh_latent = mesh_latent.squeeze(1)
                if ref_latent.dim() == 5:
                    ref_latent = ref_latent.squeeze(1)
                
                # Ensure we have exactly 4 dimensions
                if mesh_latent.dim() != 4:
                    raise ValueError(f"Mesh latent has wrong number of dimensions: {mesh_latent.dim()}, shape: {mesh_latent.shape}")
                if ref_latent.dim() != 4:
                    raise ValueError(f"Ref latent has wrong number of dimensions: {ref_latent.dim()}, shape: {ref_latent.shape}")
                
                debug_print(f"Final generated latent shapes - mesh: {mesh_latent.shape}, ref: {ref_latent.shape}")
                
                # Save the latents
                self.mesh_latent_dir.mkdir(parents=True, exist_ok=True)
                self.ref_latent_dir.mkdir(parents=True, exist_ok=True)
                torch.save(mesh_latent.cpu(), self.mesh_latent_dir / f"{frame_id}.pt")
                torch.save(ref_latent.cpu(), self.ref_latent_dir / f"{frame_id}.pt")
        
        return self.tf(render_img), self.tf(face_img), mesh_latent, ref_latent

# ─── 3) HELPER FUNCTIONS ──────────────────────────────────────────────────────

def get_image_transform(resolution, dtype=torch.float32):
    """Return a torchvision transform for image preprocessing.
    
    This transform:
    1. Resizes the image to the target resolution
    2. Converts to tensor in [0,1] range
    3. Normalizes to [-1,1] range (which is what SD expects)
    4. Converts to the specified dtype
    """
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),                    # [0,1]
        T.Normalize([0.5]*3, [0.5]*3),   # → [-1,1]
        T.Lambda(lambda x: x * 0.5 + 0.5),
        T.ConvertImageDtype(dtype),
    ])

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to a PIL Image.
    
    Args:
        x: PyTorch tensor [3,H,W] with values in [0,1] or [-1,1]
        
    Returns:
        PIL Image in RGB format
    """
    # Convert to numpy and ensure proper range
    if x.min() < 0:  # If in [-1,1] range
        x = (x.clamp(-1,1) + 1) / 2
    x = x.clamp(0,1).cpu().permute(1,2,0).numpy()
    return Image.fromarray((x * 255).astype(np.uint8))

def pil_to_tensor(x: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor.
    
    Args:
        x: PIL Image in RGB format
        device: Device to place tensor on
        
    Returns:
        PyTorch tensor [1,3,H,W] with values in [-1,1] range
    """
    # Convert to tensor and normalize to [-1,1]
    x = torch.from_numpy(np.array(x)).float() / 255.0  # [H,W,3] in [0,1]
    x = x.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    x = (x * 2 - 1).to(device)  # Convert to [-1,1] range
    return x

def preencode_latents(pipe, tf, character_path, device="cuda", resolution=None):
    """Precompute and save latents for all mesh and face images."""
    vae = pipe.vae.eval()

    # Use MeshFaceDataset to get the exact same data
    dataset = MeshFaceDataset(character_path, tf, resolution, vae)

    # Create resolution-specific subdirectories
    res_suffix = f"_{resolution}" if resolution else ""
    mesh_latent_dir = character_path / "processed" / "latents" / f"mesh{res_suffix}"
    ref_latent_dir = character_path / "processed" / "latents" / f"ref{res_suffix}"
    
    # Remove existing latents to ensure clean state
    if mesh_latent_dir.exists():
        for f in mesh_latent_dir.glob("*.pt"):
            f.unlink()
    if ref_latent_dir.exists():
        for f in ref_latent_dir.glob("*.pt"):
            f.unlink()
            
    # Create fresh directories
    mesh_latent_dir.mkdir(parents=True, exist_ok=True)
    ref_latent_dir.mkdir(parents=True, exist_ok=True)

    # Process pairs
    with torch.no_grad():
        for i in range(len(dataset)):
            render_img, face_img, _, _ = dataset[i]
            frame_id = dataset.frame_ids[i]
            
            mesh_latent_path = mesh_latent_dir / f"{frame_id}.pt"
            ref_latent_path = ref_latent_dir / f"{frame_id}.pt"

            # Process images
            render_img = render_img.unsqueeze(0).to(device)
            face_img = face_img.unsqueeze(0).to(device)

            # Encode to latents and ensure correct dimensions
            mesh_latent = vae.encode(render_img).latent_dist.sample() * vae.config.scaling_factor
            ref_latent = vae.encode(face_img).latent_dist.sample() * vae.config.scaling_factor
            
            # Ensure latents are 4D [B, C, H, W]
            if mesh_latent.dim() == 5:
                mesh_latent = mesh_latent.squeeze(1)
            if ref_latent.dim() == 5:
                ref_latent = ref_latent.squeeze(1)

            # Save latents
            torch.save(mesh_latent, mesh_latent_path)
            torch.save(ref_latent, ref_latent_path)
            debug_print(f"Saved latents for {frame_id} at resolution {resolution}")

    # Update the dataset to use the new latent directories
    dataset.mesh_latent_dir = mesh_latent_dir
    dataset.ref_latent_dir = ref_latent_dir
    dataset.frame_ids = [p.stem for p in dataset.renders]  # Ensure frame IDs match the renders
    
    return dataset  # Return the updated dataset

def latent_to_image(pipe, latents, device, batch_idx=0, return_type="pil"):
    """Convert latents to an image format.
    
    Args:
        pipe: The SDXL pipeline with VAE
        latents: Input latents tensor [B,C,H,W]
        device: Device to perform computation on
        batch_idx: Index of the batch to convert (default: 0)
        return_type: One of ["pil", "tensor", "numpy", "tensor_hwc"] to specify return format
                    - "pil": PIL Image
                    - "tensor": torch.Tensor in [0,255] range, shape [B,3,H,W]
                    - "tensor_hwc": torch.Tensor in [0,255] range, shape [B,H,W,3]
                    - "numpy": numpy array in [0,255] range
        
    Returns:
        Image in the requested format
    """
    # Ensure latents are 4D
    if latents.dim() == 5:
        latents = latents.squeeze(1)
    
    # Ensure VAE and latents are in the same dtype
    latents = latents.to(pipe.vae.dtype)
    
    # Decode latents
    decoded = pipe.vae.decode(latents).sample  # [B,3,H,W]

    # Check if we need to scale from [-1,1] to [0,1]
    if round(decoded.min().item()) < 0:
        decoded = (decoded + 1) / 2
    
    # Scale to [0,255] and clamp
    decoded = (decoded * 255).clamp(0, 255)
    
    if return_type == "tensor":
        # Return in [B,3,H,W] format
        return decoded.to(device)
    elif return_type == "tensor_hwc":
        # Return in [B,H,W,3] format
        return decoded.permute(0, 2, 3, 1).to(device)
    else:  # "pil"
        # Convert single image to PIL
        img = decoded[batch_idx]  # [3,H,W]
        return Image.fromarray(img.byte().permute(1,2,0).cpu().numpy())

def save_training_preview(pipe, batch, output_dir, step, device, config):
    """Save a training preview with mesh, reference, and generated images."""
    try:
        # Create preview directory
        preview_dir = Path(output_dir) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"preview_{step:06d}.png"
        
        # Extract inputs
        mesh_imgs, ref_imgs, mesh_latents, _ = batch
        
        # Switch to eval mode for inference
        pipe.unet.eval()
        
        with torch.no_grad():
            # Convert BFloat16 to Float32 to avoid dtype issues
            mesh_imgs = mesh_imgs.to(device).to(torch.float32)
            ref_imgs = ref_imgs.to(device).to(torch.float32)
            
            # Generate with pipe using text prompt from config
            out = pipe(
                image=mesh_imgs[:1],
                prompt=config["diffusion"]["prompt"],
                num_inference_steps=config["diffusion"]["num_timesteps"],
                guidance_scale=config["diffusion"]["guidance_scale"],
                gradient=False
            ).images[0]
            
            # For debug purposes, if we want to visualize how the latents decode directly
            global DEBUG_MODE
            if mesh_latents is not None and DEBUG_MODE:
                # Convert latents to image
                img = latent_to_image(pipe, mesh_latents, device)
                    
                # Save for debugging
                debug_dir = Path(output_dir) / "latent_debug"
                debug_dir.mkdir(exist_ok=True, parents=True)
                img.save(debug_dir / f"decoded_latent_{step:06d}.png")
            
            # Convert tensors to PIL
            mesh_pil = tensor_to_pil(mesh_imgs[0])
            ref_pil = tensor_to_pil(ref_imgs[0])
            
            # Create grid with PIL
            W, H = mesh_pil.size
            grid = Image.new('RGB', (W * 3, H))
            grid.paste(mesh_pil, (0, 0))
            grid.paste(ref_pil, (W, 0))
            grid.paste(out, (W * 2, 0))
            
            # Save grid
            grid.save(preview_path)
        print(f"✅ Saved training preview to {preview_path}")
            
    except Exception as e:
        print(f"Error in save_training_preview: {e}")
        import traceback
        print(traceback.format_exc())
        print("Continuing training without preview...")
    finally:
        # Always switch back to train mode
        pipe.unet.train()

def to_float(val):
    """Safely convert a tensor or float to a Python float."""
    try:
        if torch.is_tensor(val):
            return val.item()
        return float(val)
    except:
        return 0.0
    
def sample_timesteps_linear(epoch, config, noise_scheduler, batch_size, device):
    """
    Linearly increase available timesteps by `step_frac` each epoch.
    At epoch=0 you get `step_frac`*T, at epoch=1 you get 2*step_frac*T, etc.
    Once you hit 100%, you stay there.
    """
    T = noise_scheduler.config.num_train_timesteps
    # how much to grow per epoch (e.g. 0.1 for 10% per epoch)
    step_frac = config["schedule"]["spatial"].get("curriculum_step_frac", 0.1)

    # compute fraction of the full schedule to allow this epoch
    frac = min(1.0, (epoch + 1) * step_frac)
    # compute max index (at least 1)
    t_max = max(1, int(frac * T))

    # sample uniformly from [0, t_max)
    return torch.randint(0, t_max, (batch_size,), device=device)

def forward_pass(
    pipe,
    batch,
    noise_scheduler,
    seq_embeds,
    pooled_embeds,
    device,
    epoch: int,
    config: dict,
):
    """Perform a forward pass through the UNet.
    
    Args:
        pipe: The SDXL pipeline
        batch: Tuple of (mesh_imgs, ref_imgs, mesh_latents, ref_latents)
        noise_scheduler: The noise scheduler
        seq_embeds: Sequence embeddings [1, seq_len, hidden_dim]
        pooled_embeds: Pooled embeddings [1, hidden_dim]
        device: Device to use
        epoch: Current training epoch
        config: Configuration dictionary
        
    Returns:
        tuple: (noise_pred, noise, timesteps, noisy_ref, mesh_imgs, ref_imgs, mesh_latents, ref_latents)
    """
    # Unpack batch
    mesh_imgs, ref_imgs, mesh_latents, ref_latents = batch
    
    # Ensure latents are 4D before moving to device
    if mesh_latents.dim() == 5:
        mesh_latents = mesh_latents.squeeze(1)
    if ref_latents.dim() == 5:
        ref_latents = ref_latents.squeeze(1)
    
    # Move to device
    mesh_imgs = mesh_imgs.to(device)
    mesh_latents = mesh_latents.to(device)
    ref_latents = ref_latents.to(device).detach()
    ref_imgs = ref_imgs.to(device).detach()
    
    # Add noise to ref latents
    noise = torch.randn_like(ref_latents)
    timesteps = sample_timesteps_linear(epoch, config, noise_scheduler, mesh_latents.shape[0], device)
    noisy_ref = noise_scheduler.add_noise(ref_latents, noise, timesteps)
        
    # expand prompt embeddings for both conditioned and unconditioned
    bsz = ref_latents.shape[0]
    seq = seq_embeds.repeat(bsz, 1, 1).to(device).detach()
    pool = pooled_embeds.repeat(bsz, 1).to(device).detach()
    
    # Get image dimensions from mesh_imgs
    H, W = mesh_imgs.shape[-2:]
    
    # Create add_time_ids
    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size=(H, W),
        crops_coords_top_left=(0, 0),
        target_size=(H, W),
        aesthetic_score=0.0,
        negative_aesthetic_score=0.0,
        negative_original_size=(H, W),
        negative_crops_coords_top_left=(0, 0),
        negative_target_size=(H, W),
        dtype=seq_embeds.dtype,
        text_encoder_projection_dim=pooled_embeds.shape[-1],
    )
    add_time_ids = add_time_ids.repeat(bsz, 1).to(device)
    add_neg_time_ids = add_neg_time_ids.repeat(bsz, 1).to(device)
        
    # Run UNet forward pass
    noise_pred = pipe.unet(
        noisy_ref,
        timesteps,
        encoder_hidden_states=seq,
        added_cond_kwargs={
            "orig_image_latents": mesh_latents,
            "text_embeds": pool,
            "time_ids": add_time_ids,
            "neg_time_ids": add_neg_time_ids
        },
    ).sample
    
    return noise_pred, noise, timesteps, noisy_ref, mesh_imgs, ref_imgs, mesh_latents, ref_latents

def calculate_mse_patch_loss(noise_pred, noise, alpha_bar, config):
    """Calculate MSE and patch losses.
    
    Args:
        noise_pred: Predicted noise [B,C,H,W]
        noise: Target noise [B,C,H,W]
        alpha_bar: Alpha bar values for current timesteps [B,1,1,1]
        config: Configuration dictionary
        
    Returns:
        tuple: (mse_loss, patch_loss)
    """
    # Calculate signal-to-noise ratio weights (1/√(ᾱ_t))
    snr       = torch.sqrt(alpha_bar / (1 - alpha_bar))
    
    # Weighted MSE loss
    mse_loss = F.mse_loss(noise_pred, noise) #(snr * (noise_pred - noise) ** 2).mean()
    
    # Patch loss
    H, W = noise_pred.shape[-2:]
    ps = int(H * config["loss_spatial"]["patch_ratio"])
    
    # Calculate patch loss from 5 random locations
    patch_losses = []
    for _ in range(5):
        y, x = torch.randint(0, H-ps+1, (1,)), torch.randint(0, W-ps+1, (1,))
        p_pred = noise_pred[:, :, y:y+ps, x:x+ps]
        p_gt = noise[:, :, y:y+ps, x:x+ps]
        patch_losses.append(F.mse_loss(p_pred, p_gt))
    
    # Average the patch losses
    patch_loss = torch.stack(patch_losses).mean()
    
    return mse_loss, patch_loss

def calculate_perceptual_loss(pipe, latents, ref_latents, vgg, lpips_loss_fn, device):
    """Calculate perceptual losses using VGG and LPIPS.
    
    Args:
        pipe: The SDXL pipeline
        latents: Predicted latents [B,C,H,W]
        ref_latents: Target latents [B,C,H,W]
        vgg: VGG model for perceptual features (expects [0,1] range)
        lpips_loss_fn: LPIPS loss function (expects [-1,1] range)
        device: Device to use
        
    Returns:
        float: Combined perceptual loss
    """
    try:
        # Ensure latents are in the same dtype as VAE before decoding
        latents_for_decode = latents.to(pipe.vae.dtype)
        ref_latents_for_decode = ref_latents.to(pipe.vae.dtype)
        
        # Decode with proper scaling
        decoded = pipe.vae.decode(latents_for_decode).sample
        decoded_target = pipe.vae.decode(ref_latents_for_decode).sample
        
        # Convert to float32 for VGG and LPIPS
        decoded = decoded.to(torch.float32)
        decoded_target = decoded_target.to(torch.float32)
        
        # Resize to 224x224 for VGG
        vgg_input_pred = F.interpolate(decoded, size=(224, 224), mode='bilinear', align_corners=False)
        vgg_input_target = F.interpolate(decoded_target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Check if we need to scale to [0,1] for VGG
        if round(vgg_input_pred.min().item()) < 0:
            vgg_input_pred = (vgg_input_pred + 1) / 2
            vgg_input_target = (vgg_input_target + 1) / 2

        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

        vgg_input_pred = normalize(vgg_input_pred.clamp(0, 1))
        vgg_input_target = normalize(vgg_input_target.clamp(0, 1))
        
        # Extract VGG features
        f_pred = vgg(vgg_input_pred)
        f_ref = vgg(vgg_input_target)
        
        # VGG perceptual loss (L1)
        vgg_loss = F.l1_loss(f_pred, f_ref)
        
        # Check if we need to scale to [-1,1] for LPIPS
        if round(decoded.min().item()) >= 0:  # If in [0,1] range
            pred_n = decoded * 2 - 1
            tgt_n = decoded_target * 2 - 1
        else:  # Already in [-1,1] range
            pred_n = decoded
            tgt_n = decoded_target
        
        # Ensure [-1,1] range for LPIPS
        pred_n = pred_n.clamp(-1, 1)
        tgt_n = tgt_n.clamp(-1, 1)
        
        # Calculate LPIPS loss
        perc_map = lpips_loss_fn(pred_n, tgt_n)
        lpips_loss = perc_map.mean()
        
        # Combine both perceptual losses
        return vgg_loss + lpips_loss
        
    except Exception as e:
        print(f"Error in perceptual loss calculation: {e}")
        return torch.tensor(0.0, device=device)
    
def calculate_id_loss(pipe, fake, real, resnet, mtcnn, device):
    # faces_fake: [B,3,160,160], same for faces_real

    # Scale images to 160x160 for face detection
    fake = F.interpolate(fake, size=(160, 160), mode='bilinear', align_corners=False)
    real = F.interpolate(real, size=(160, 160), mode='bilinear', align_corners=False)
    # Permute from [B,H,W,3] to [B,3,H,W] for interpolation
    fake = fake.permute(0, 2, 3, 1)
    real = real.permute(0, 2, 3, 1)

    try:
        faces_real = mtcnn(real.cpu())
        if faces_real is None or (isinstance(faces_real, (list, tuple)) and len(faces_real) == 0):
            print("⚠️ No faces detected in real image — skipping this batch")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in real face detection: {e}")
        return torch.tensor(2., device=device)
    
    
    try:
        faces_fake = mtcnn(fake.cpu())
        if faces_fake is None or (isinstance(faces_fake, (list, tuple)) and len(faces_fake) == 0):
            print("⚠️ No faces detected in fake image — skipping this batch")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in fake face detection: {e}")
        return torch.tensor(2., device=device)
    debug_print(faces_fake)
    debug_print(faces_real)
    
    # guard
    if faces_fake == [None] or faces_real == [None]:
        # either None or empty list
        print("Skipping ID loss: insufficient faces")
        return torch.tensor(2., device=device)

    # list → tensor
    faces_fake = torch.cat([f.unsqueeze(0) for f in faces_fake], dim=0).to(device)
    faces_real = torch.cat([f.unsqueeze(0) for f in faces_real], dim=0).to(device)
    #print("faces_fake shape:", faces_fake.shape)
    #print("faces_real shape:", faces_real.shape)

    # 3) Get embeddings
    #    These are [B,512] L2-normalized by the model
    try:
        emb_fake = resnet(faces_fake)            # requires_grad=True
        with torch.no_grad():
            emb_real = resnet(faces_real)        # detach real to block grads back to data
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in embedding calculation: {e}")
        return torch.tensor(2., device=device)
    debug_print(emb_fake)
    debug_print(emb_real)

    # 4) Compute identity loss (1 − cosine similarity)
    #    F.cosine_similarity returns [B], so we take the mean
    cos_sim = F.cosine_similarity(emb_fake, emb_real, dim=1)  # in [-1,1], 1=perfect match
    id_loss = (1 - cos_sim).mean()                            # scalar loss

    return id_loss

def spatial_step(
    pipe,
    batch,
    noise_scheduler,
    epoch: int,
    step_idx: int,
    config: dict,
    device: str,
    accelerator,
    optimizer_G,
    optimizer_D,
    lr_scheduler,
    unet_ema,
    lpips_loss_fn,
    disc,
    seq_embeds,
    pooled_embeds,
    vgg,
    mtcnn,
    resnet,
    is_training: bool = True,
    output_dir: Optional[Path] = None,
):
    """Main spatial training step function."""
    debug_print(f"\n=== Step {step_idx} Debug Info ===")
    debug_print(f"Training mode: {is_training}")
    debug_print(f"Epoch: {epoch}")
    debug_print(f"GAN active: {epoch >= config['schedule']['spatial']['gan_start']}")
    
    # Initialize tracking dictionaries
    losses = {}
    
    # Forward pass - wrapped in no_grad for validation
    context = torch.no_grad() if not is_training else nullcontext()
    with context:
        noise_pred, noise, timesteps, noisy_ref, mesh_imgs, ref_imgs, mesh_latents, ref_latents = forward_pass(
            pipe, batch, noise_scheduler,
            seq_embeds, pooled_embeds, device,
            epoch, config
        )
        
        # Get alpha_bar for current timesteps
        alpha_bar = noise_scheduler.alphas_cumprod[timesteps.cpu()].view(-1, 1, 1, 1).to(device)
        
        # Calculate MSE and patch losses
        mse_loss, patch_loss = calculate_mse_patch_loss(noise_pred, noise, alpha_bar, config)
        
        # Initialize total loss
        total_loss = (
            config["loss_spatial"]["lambda_mse"] * mse_loss +
            config["loss_spatial"]["lambda_patch"] * patch_loss
        )
        
        # Get denoised latents for GAN if needed
        fake_latents = None
        if is_training and epoch >= config["schedule"]["spatial"]["gan_start"] and step_idx % config["loss_spatial"]["gan_frequency"] == 0:
            out = pipe(
                image=mesh_imgs,
                prompt=config["diffusion"]["prompt"],
                num_inference_steps=config["diffusion"]["num_timesteps"],
                guidance_scale=config["diffusion"]["guidance_scale"],
                output_type="latent",
                gradient=True
            )
            fake_latents = out["images"] / pipe.vae.config.scaling_factor
            ref_latents_for_decode = ref_latents / pipe.vae.config.scaling_factor

            # Save debug preview if output_dir is provided
            if output_dir is not None and step_idx % 2 == 0:
                with torch.no_grad():
                    debug_print("Saving debug preview")
                    preview_dir = Path(output_dir) / "latent_debug"
                    preview_dir.mkdir(exist_ok=True, parents=True)
                    preview_path = preview_dir / f"latent_preview_{step_idx}.png"
                    
                    # Create a grid showing both real and fake images
                    W, H = 512, 512  # Assuming 512x512 images
                    grid = Image.new('RGB', (W * 2, H))
                    
                    fake_pil = latent_to_image(pipe, fake_latents, device, 0, return_type="pil")
                    real_pil = latent_to_image(pipe, ref_latents_for_decode, device, 0, return_type="pil")
                    
                    # Add labels
                    from PIL import ImageDraw
                    draw_fake = ImageDraw.Draw(fake_pil)
                    draw_real = ImageDraw.Draw(real_pil)
                    draw_fake.text((10, 10), "Fake", fill=(255, 255, 255))
                    draw_real.text((10, 10), "Real", fill=(255, 255, 255))
                    
                    grid.paste(fake_pil, (0, 0))
                    grid.paste(real_pil, (W, 0))
                    
                    # Save the grid
                    grid.save(preview_path)
                    debug_print(f"Saved latent preview to {preview_path}")
            
            # Convert both real and fake latents to tensors in [0,255] range
            fake = latent_to_image(pipe, fake_latents, device, return_type="tensor")  # [B,3,H,W] for discriminator
            real = latent_to_image(pipe, ref_latents_for_decode, device, return_type="tensor")   # [B,3,H,W] for discriminator
        
        # Add perceptual loss if applicable
        perc_loss = torch.tensor(0.0, device=device)
        if epoch >= config["schedule"]["spatial"]["perceptual_start"] and step_idx % config["loss_spatial"]["perceptual_frequency"] == 0:
            # Reconstruct predicted clean sample x0_pred using alpha_bar
            x0_pred = (noisy_ref - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
            # Compute perceptual loss
            perc_loss = calculate_perceptual_loss(
                pipe, x0_pred.to(device), ref_latents.to(device), vgg, lpips_loss_fn, device
            )
            total_loss = total_loss + config['loss_spatial']['lambda_perceptual'] * perc_loss
    
    # Training-specific updates
    if is_training:
        # Discriminator update if applicable
        if epoch >= config["schedule"]["spatial"]["gan_start"] and step_idx % config["loss_spatial"]["gan_frequency"] == 0:
            # Calculate ID loss
            id_loss = calculate_id_loss(pipe, fake, real, resnet, mtcnn, device)  # Uses [B,H,W,3]
            total_loss += config["loss_spatial"]["lambda_id"] * id_loss
            losses["id_loss"] = id_loss.item()

            debug_print("\n[Discriminator Update]")
            
            criterion_mse = nn.MSELoss()
            fake = fake.to(torch.float32).div(255.0)   # → [0,1]
            real = real.to(torch.float32).div(255.0)   # → [0,1]

            # Discriminator step
            logits_real = disc(real)                  # ∈ [0,1]
            logits_fake = disc(fake.detach())         # ∈ [0,1]

            real_labels = torch.ones_like(logits_real)  # all 1s
            fake_labels = torch.zeros_like(logits_fake) # all 0s

            loss_D_real = criterion_mse(logits_real, real_labels)  # ∈ [0,1]
            loss_D_fake = criterion_mse(logits_fake, fake_labels)  # ∈ [0,1]
            loss_D      = 0.5*(loss_D_real + loss_D_fake)          # ∈ [0,1]
            
            # Update discriminator
            optimizer_D.zero_grad(set_to_none=True)
            accelerator.backward(loss_D)
            optimizer_D.step()
            optimizer_D.zero_grad()
            
            losses.update({
                "loss_D": loss_D.item(),
                "loss_D_real": loss_D_real.item(),
                "loss_D_fake": loss_D_fake.item(),
            })
            
            # Generator step
            # wants D(fake) → 1
            loss_G_adv = criterion_mse(disc(fake), real_labels)  # ∈ [0,1]
            total_loss += config["loss_spatial"]["lambda_adv"] * loss_G_adv
            losses["loss_G_adv"] = loss_G_adv.item()
        else:
            losses.update({
                "loss_D": 0.0,
                "loss_D_real": 0.0,
                "loss_D_fake": 0.0,
                "loss_G_adv": 0.0,
                "id_loss": 0.0,
            })
        
        # Generator update
        debug_print("\n[Generator Update]")
        optimizer_G.zero_grad(set_to_none=True)
        accelerator.backward(total_loss)
        optimizer_G.step()
        lr_scheduler.step()
        optimizer_G.zero_grad()
        
        if unet_ema is not None and accelerator.sync_gradients:
            unet_ema.step(pipe.unet.parameters())
    
    # Store all losses
    losses.update({
        "loss": total_loss.item(),
        "mse_loss": mse_loss.item(),
        "patch_loss": patch_loss.item(),
        "perc_loss": perc_loss.item(),
    })
    
    return losses

# ─── 4) TRAINING LOOP ──────────────────────────────────────────────────────────
def train_lora(character_name: str, output_dir: Optional[str] = None, resume_from: Optional[str] = None):
    print("\n=== Starting training setup ===")
    
    # ─── Stage 0: Initialize paths and config ────────────────────
    output_dir = Path(output_dir) if output_dir else Path(DATA_MOUNT) / character_name / "trainings" / f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nTraining output will be saved to: {output_dir}")
    
    # Load config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize DEBUG_MODE from config
    global DEBUG_MODE
    DEBUG_MODE = config.get("debug", {}).get("enabled", False)
    debug_print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
        
    # ─── Stage 1: Setup hardware and acceleration ────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(
        mixed_precision=config["optimization"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
    )

    # ─── Stage 2: Load base model and setup LoRA ─────────────────
    # Load SD-XL pipeline
    modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
    sdxl_cache_path = os.path.join(modal_cache, "huggingface", "sdxl-base-1.0")
        
    pipe = GradStableDiffusionXLImg2ImgPipeline.from_pretrained(
        sdxl_cache_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )

    # Configure LoRA adapter
    lora_cfg = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=[
            "conv1", "conv2", "conv_shortcut", "to_q", "to_k", "to_v", "proj_in", "proj_out"
        ],
        bias="none",
    )

    # ─── Stage 3: Load from checkpoint (if requested) ───────────
    start_epoch = 0
    global_step = 0
    checkpoint_state = None

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        pipe.unet, checkpoint_state = load_checkpoint(resume_from, pipe.unet, lora_cfg=lora_cfg, device="cpu")
        pipe.unet = pipe.unet.to(torch.bfloat16)
        
        # Load training state
        if "epoch" in checkpoint_state:
            start_epoch = checkpoint_state["epoch"]
        if "step" in checkpoint_state:
            global_step = checkpoint_state["step"]
    else:
        # Apply LoRA adapter to UNet
        pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    # ─── Stage 4: Setup model for training ──────────────────────
    # Initialize EMA model AFTER LoRA is applied to ensure matching architecture
    unet_ema = EMAModel(
        pipe.unet.parameters(),
        decay=0.9999,
        model_cls=pipe.unet.__class__,
        device="cpu"
    )
    
    # Load EMA state from checkpoint if present
    if resume_from and checkpoint_state and "ema" in checkpoint_state:
        print("Loading EMA state from checkpoint")
        unet_ema.load_state_dict(checkpoint_state["ema"])
    
    # Move EMA to device
    unet_ema.to(device)
    
    # Performance optimizations
    pipe.enable_attention_slicing()
    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.eval()
    for p in pipe.vae.parameters():
        p.requires_grad = False

    # ─── Stage 5: Prepare datasets and dataloaders ─────────────
    # Character data paths
    data_mount = Path(DATA_MOUNT)
    character_path = data_mount / character_name

    # Progressive resolution setup
    full_res = config["data"]["full_resolution"]                    
    init_res = config["data"]["initial_resolution"]                        
    switch_ep = config["data"].get("progressive_switch_epoch", 10)
    
    # Initialize with low resolution
    initial_transform = get_image_transform(init_res, dtype=torch.bfloat16)
    
    # Precompute latents for initial resolution (we'll compute full res later when needed)
    dataset = preencode_latents(pipe, initial_transform, character_path, device, resolution=init_res)
    
    # Setup main dataset with initial resolution
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Set transforms for both dataloaders
    train_loader.dataset.dataset.tf = initial_transform
    val_loader.dataset.dataset.tf = initial_transform
    
    # Setup temporal dataset (optional)
    video_character_name = "documale1"
    mesh_dir = Path(DATA_MOUNT) / video_character_name / "processed" / "renders"
    flow_path = Path(DATA_MOUNT) / video_character_name / "flows.npz"
    video_ds = TemporalDataset(mesh_dir, flow_path, initial_transform)
    video_loader = DataLoader(video_ds, batch_size=1, shuffle=True)

    # ─── Stage 6: Create optimizers and schedulers ─────────────
    # Generator optimizer (UNet)
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    debug_print(">> Config LR:", config["training"]["learning_rate"])
    debug_print(">> Param-group init:", optimizer.param_groups[0]["lr"])
    
    # Discriminator optimizer
    disc = PatchDiscriminator(in_channels=3).to(device)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=float(config["training"]["learning_rate"]))

    # Learning rate scheduler
    num_training_steps = len(train_loader) * config["schedule"]["spatial"]["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Load optimizer/scheduler state if resuming
    if resume_from and checkpoint_state:
        if "optimizer" in checkpoint_state:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        if "scheduler" in checkpoint_state:
            lr_scheduler.load_state_dict(checkpoint_state["scheduler"])
    
    # Noise scheduler for diffusion
    noise_scheduler = DDPMScheduler.from_pretrained(
        sdxl_cache_path,
        subfolder="scheduler"
    )
    noise_scheduler.set_timesteps(config["diffusion"]["num_timesteps"])

    # ─── Stage 7: Setup loss functions and perceptual models ───
    mtcnn = MTCNN(image_size=160, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # LPIPS perceptual loss - initialized safely for our use case
    lpips_loss_fn = lpips.LPIPS(net='vgg', spatial=False).to(device)
    # Ensure it's in eval mode and doesn't track gradients
    lpips_loss_fn.eval()
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    # Initialize VGG model
    vgg = torchvision.models.vgg16(pretrained=True).features.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    
    # ─── Stage 8: Final preparation for training ──────────────
    # Text embeddings for conditioning - create single prompt embedding
    prompt = ["<rrrdaniel>"]
    seq_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # Prepare all models with accelerator
    optimizer, train_loader, val_loader, lr_scheduler, video_loader, disc, opt_disc, lpips_loss_fn, noise_scheduler, mtcnn, resnet = accelerator.prepare(
        optimizer, train_loader, val_loader, lr_scheduler, video_loader, disc, opt_disc, lpips_loss_fn, noise_scheduler, mtcnn, resnet
    )
    
    # ───────────────────────────────────────────────── #
    # ───────────────── TRAINING LOOP ───────────────── #
    # ───────────────────────────────────────────────── #

    # ─── A) SPATIAL TRAINING ─────────────────────────────────────────────
    best_spatial, spatial_patience = float('inf'), 0
    
    try:
        prev_res = init_res
        for epoch in range(start_epoch, config["schedule"]["spatial"]["num_epochs"]):
            # progressive resize
            cur_res = init_res if epoch < switch_ep else full_res
            cur_tf  = get_image_transform(cur_res, dtype=torch.bfloat16)
            
            # Check if we need to switch resolution and precompute latents
            if epoch == 0 or (epoch == switch_ep and cur_res != prev_res):
                # Precompute latents for this resolution if not already done
                latent_path = character_path / "processed" / "latents" / f"mesh_{cur_res}"
                if not latent_path.exists():
                    print(f"Precomputing latents for resolution {cur_res}")
                    preencode_latents(pipe, cur_tf, character_path, device, resolution=cur_res)
                
                # Update dataset to use the new resolution latents
                train_loader.dataset.dataset.update_resolution(cur_res)
                val_loader.dataset.dataset.update_resolution(cur_res)
            
            # Reset spatial patience when new components are activated
            if epoch == config["schedule"]["spatial"]["perceptual_start"]:
                print("Activating perceptual losses - resetting spatial patience")
                spatial_patience = 0
            
            if epoch == config["schedule"]["spatial"]["gan_start"]:
                print("Activating GAN components - resetting spatial patience")
                spatial_patience = 0
            
            # Update the transforms
            train_loader.dataset.dataset.tf = cur_tf
            val_loader.dataset.dataset.tf   = cur_tf
            video_loader.dataset.tf = cur_tf

            prev_res = cur_res
            # spatial training epoch
            pipe.unet.train()
            total_loss = 0.0
            log_steps = config["debug"]["log_steps"]
            preview_steps = config["debug"]["preview_steps"]

            for step, batch in enumerate(train_loader):
                # Extra safety: clear memory and reset anything that might be retained between steps
                torch.cuda.empty_cache()
                gc.collect()
                    
                with accelerator.accumulate():
                    # Forward pass & loss computation
                    try:
                        losses = spatial_step(
                            pipe, batch, noise_scheduler,
                            epoch, global_step, config,
                            device, accelerator,
                            optimizer, opt_disc, lr_scheduler,
                            unet_ema,
                            lpips_loss_fn, disc,
                            seq_embeds, pooled_embeds,
                            vgg,
                            mtcnn,
                            resnet,
                            is_training=True,
                            output_dir=output_dir
                        )
                        
                        # Step counters
                        loss = losses["loss"]
                        total_loss += loss
                                
                    except Exception as e:
                        print(f"Error in training step: {e}")
                        # Clear any remaining computational graph on error
                        optimizer.zero_grad(set_to_none=True)
                        opt_disc.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        raise e

                # Logging
                if global_step % log_steps == 0:
                    # Calculate weighted losses
                    weighted_mse = config["loss_spatial"]["lambda_mse"] * losses['mse_loss']
                    weighted_patch = config["loss_spatial"]["lambda_patch"] * losses['patch_loss']
                    weighted_perceptual = config["loss_spatial"]["lambda_perceptual"] * losses.get('perc_loss', 0.0)
                    weighted_gan_g = config["loss_spatial"]["lambda_adv"] * losses.get('loss_G_adv', 0.0)
                    
                    accelerator.print(
                        f"[Spatial] Step {global_step} | LR {optimizer.param_groups[0]['lr']:.5f} | "
                        f"Loss {loss:.4f} | "
                        f"MSE {losses['mse_loss']:.4f} (w: {weighted_mse:.4f}) | "
                        f"Patch {losses['patch_loss']:.4f} (w: {weighted_patch:.4f}) | "
                        f"Perceptual {losses.get('perc_loss', 0.0):.4f} (w: {weighted_perceptual:.4f}) | "
                        f"GAN G {losses.get('loss_G_adv', 0.0):.4f} (w: {weighted_gan_g:.4f}) | "
                        f"GAN D {losses.get('loss_D', 0.0):.4f} | "
                        f"ID {losses.get('id_loss', 0.0):.4f}"
                    )

                # Save preview images every so often
                if global_step % preview_steps == 0:
                    save_training_preview(
                        pipe, batch, 
                        output_dir, 
                        global_step, 
                        device, 
                        config
                    )
                    
                global_step += 1

            # Validation - moved outside the training loop
            pipe.unet.eval()

            # Initialize accumulators
            agg = { k: 0.0 for k in [
                "loss", "mse_loss", "perc_loss", "patch_loss", "loss_D", "loss_G_adv"
            ]}
            n = 0
            
            # Store training weights and load EMA weights for validation
            if unet_ema is not None:
                unet_ema.store(pipe.unet.parameters())
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    losses = spatial_step(
                        pipe, batch, noise_scheduler,
                        epoch, step, config,
                        device, accelerator,
                        optimizer, opt_disc, lr_scheduler,
                        unet_ema,
                        lpips_loss_fn, disc,
                        seq_embeds, pooled_embeds,
                        vgg,
                        mtcnn,
                        resnet,
                        is_training=False,
                        output_dir=output_dir
                    )
                    # accumulate
                    for k in agg:
                        agg[k] += losses.get(k, 0.0)
                    n += 1

            # average
            for k in agg:
                agg[k] /= max(n, 1)
            
            # Print validation metrics
            accelerator.print(
                f"[Validation] Epoch {epoch} | Loss {agg['loss']:.4f} | "
                f"MSE {agg['mse_loss']:.4f} | Perceptual {agg.get('perc_loss', 0.0):.4f} | "
                f"ID {agg.get('id_loss', 0.0):.4f} | Patience {spatial_patience}/{config['training']['spatial_early_stop_patience']}"
            )
            
            # Early stopping
            if agg["loss"] < best_spatial:
                best_spatial, spatial_patience = agg["loss"], 0
                # Save best model
                if accelerator.is_main_process:
                    save_lora_state(
                    pipe, unet_ema, output_dir / "best_model", config,
                    best_loss=best_spatial, is_best=True
                )
            else:
                spatial_patience += 1
                if spatial_patience >= config["training"]["spatial_early_stop_patience"]:
                    print("Early stopping SPATIAL")
                    break
    finally:
        # always executed, even on error or Ctrl-C
        if best_spatial < float('inf'):
            print(f"⚠️  Exiting early—saving best spatial LoRA (loss={best_spatial:.4f})")
            save_lora_state(pipe, unet_ema, output_dir/"best_model", config,
                            best_loss=best_spatial, is_best=True)
    # Save final model
    if accelerator.is_main_process:
        save_lora_state(
            pipe, unet_ema, output_dir / "final_model", config,
            is_final=True
        )

    return {
        "status": "success",
        "message": f"Training completed for {character_name}",
        "output_dir": str(output_dir)
    }