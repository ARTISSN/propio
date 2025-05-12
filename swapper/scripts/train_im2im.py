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
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torchvision import transforms as T
from PIL import Image
import yaml
from accelerate import Accelerator
import datetime
from diffusers.training_utils import EMAModel
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
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
        # Temporarily lift the no_grad guard so everything is differentiable
        with torch.enable_grad():
            return super().__call__(*args, **kwargs)

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

        # Only keep pairs where the stem is not in the blacklist
        self.renders = []
        self.faces = []
        for r, f in zip(all_renders, all_faces):
            stem = r.stem
            if stem not in blacklist:
                self.renders.append(r)
                self.faces.append(f)

        assert len(self.renders) == len(self.faces), \
            "Render and face counts do not match after applying blacklist!"
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
                mesh_latent = self.vae.encode(self.tf(render_img).unsqueeze(0)).latent_dist.sample()
                ref_latent = self.vae.encode(self.tf(face_img).unsqueeze(0)).latent_dist.sample()
                
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
            mesh_latent = vae.encode(render_img).latent_dist.sample()
            ref_latent = vae.encode(face_img).latent_dist.sample()
            
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
        return_type: One of ["pil", "tensor", "numpy"] to specify return format
                    - "pil": PIL Image
                    - "tensor": torch.Tensor in [-1,1] range
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
    with torch.no_grad():
        decoded = pipe.vae.decode(latents).sample
        img = decoded[batch_idx]  # [3,H,W] in [0,1] range
        
        if return_type == "tensor":
            # Convert to [-1,1] range
            return (img * 2 - 1).to(device)
        elif return_type == "numpy":
            # Convert to [0,255] range
            return (img * 255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
        else:  # "pil"
            # Convert to PIL Image
            return Image.fromarray((img * 255).clamp(0,255).byte().permute(1,2,0).cpu().numpy())

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
                guidance_scale=config["diffusion"]["guidance_scale"]
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

def forward_pass(
    pipe,
    batch,
    noise_scheduler,
    seq_embeds,
    pooled_embeds,
    device,
):
    """Perform a forward pass through the UNet.
    
    Args:
        pipe: The SDXL pipeline
        batch: Tuple of (mesh_imgs, ref_imgs, mesh_latents, ref_latents)
        noise_scheduler: The noise scheduler
        seq_embeds: Sequence embeddings [1, seq_len, hidden_dim]
        pooled_embeds: Pooled embeddings [1, hidden_dim]
        device: Device to use
        
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
    ref_latents = ref_latents.to(device)
    ref_imgs = ref_imgs.to(device)
    
    # Add noise to ref latents
    noise = torch.randn_like(ref_latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (mesh_latents.shape[0],), device=device)
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

def calculate_mse_patch_loss(noise_pred, noise, config):
    """Calculate MSE and patch losses.
    
    Args:
        noise_pred: Predicted noise [B,C,H,W]
        noise: Target noise [B,C,H,W]
        config: Configuration dictionary
        
    Returns:
        tuple: (mse_loss, patch_loss)
    """
    # MSE loss
    mse_loss = F.mse_loss(noise_pred, noise)
    
    # Patch loss
    H, W = noise_pred.shape[-2:]
    ps = int(H * config["loss_spatial"]["patch_ratio"])
    y, x = torch.randint(0, H-ps+1, (1,)), torch.randint(0, W-ps+1, (1,))
    p_pred = noise_pred[:, :, y:y+ps, x:x+ps]
    p_gt = noise[:, :, y:y+ps, x:x+ps]
    patch_loss = F.mse_loss(p_pred, p_gt)
    
    return mse_loss, patch_loss

def calculate_perceptual_loss(pipe, latents, ref_latents, vgg, lpips_loss_fn, device):
    """Calculate perceptual losses using VGG and LPIPS.
    
    Args:
        pipe: The SDXL pipeline
        latents: Predicted latents [B,C,H,W]
        ref_latents: Target latents [B,C,H,W]
        vgg: VGG model for perceptual features
        lpips_loss_fn: LPIPS loss function
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
        
        # Extract VGG features
        f_pred = vgg(vgg_input_pred)
        f_ref = vgg(vgg_input_target)
        
        # VGG perceptual loss (L1)
        vgg_loss = F.l1_loss(f_pred, f_ref)
        
        # Normalize to [-1,1] for LPIPS
        pred_n = decoded * 2 - 1
        tgt_n = decoded_target * 2 - 1
        
        # Calculate LPIPS loss
        perc_map = lpips_loss_fn(pred_n, tgt_n)
        lpips_loss = perc_map.mean()
        
        # Combine both perceptual losses
        return vgg_loss + lpips_loss
        
    except Exception as e:
        print(f"Error in perceptual loss calculation: {e}")
        return torch.tensor(0.0, device=device)
        
# AuraFace ID loss - temporarily disabled for training
"""
if config["loss_spatial"]["lambda_id"] > 0 and step_idx % config["loss_spatial"]["id_frequency"] == 0:
            try:
                lambda_id = config["loss_spatial"]["lambda_id"]
                debug_print(f"ID loss active: lambda_id={lambda_id}, epoch={epoch}, step={step_idx}")
                
                # Process images and get embeddings
                emb_preds = []
                emb_tgts = []
                
        for batch_idx in range(latents.shape[0]):
            try:
                # Convert latents to images using helper function with batch index
                pred_img = latent_to_image(pipe, latents, device, batch_idx)
                tgt_img = latent_to_image(pipe, ref_latents, device, batch_idx)
                        
                        if batch_idx == 0:  # only print detailed info for the first image
                    debug_print(f"AuraFace input - shape: {pred_img.shape}, dtype: {pred_img.dtype}")
                    debug_print(f"Pixel range: min={pred_img.min()}, max={pred_img.max()}, mean={pred_img.mean():.2f}")
                        
                        # Save preview of target image for debugging
                if batch_idx == 0 and output_dir is not None:
                            preview_dir = Path(output_dir) / "aura_debug"
                            preview_dir.mkdir(exist_ok=True, parents=True)
                            preview_path = preview_dir / f"aura_input_{global_step}.png"
                    
                    # Create a grid showing both images with confidence scores
                    W, H = pred_img.shape[1], pred_img.shape[0]
                    grid = Image.new('RGB', (W * 2, H))
                    
                    # Add confidence scores to images
                    pred_pil = Image.fromarray(pred_img)
                    tgt_pil = Image.fromarray(tgt_img)
                    
                    # Draw confidence scores
                    from PIL import ImageDraw
                    draw_pred = ImageDraw.Draw(pred_pil)
                    draw_tgt = ImageDraw.Draw(tgt_pil)
                    
                    # Get face detections first
                    faces_pred = aura_model.get(pred_img)
                    faces_tgt = aura_model.get(tgt_img)
                    
                    # Add confidence score text if faces are detected
                    if len(faces_pred) > 0:
                        draw_pred.text((10, 10), f"Conf: {faces_pred[0].det_score:.2f}", fill=(255, 255, 255))
                    if len(faces_tgt) > 0:
                        draw_tgt.text((10, 10), f"Conf: {faces_tgt[0].det_score:.2f}", fill=(255, 255, 255))
                    
                    grid.paste(pred_pil, (0, 0))
                    grid.paste(tgt_pil, (W, 0))
                    
                    # Save the grid
                    grid.save(preview_path)
                            debug_print(f"Saved AuraFace input preview to {preview_path}")
                        
                # Get face detections
                faces_pred = aura_model.get(pred_img)
                faces_tgt = aura_model.get(tgt_img)
                        
                debug_print(f"Detected faces - pred: {len(faces_pred)}, target: {len(faces_tgt)}")
                        
                        # Handle case where no faces are detected
                        if len(faces_pred) == 0 or len(faces_tgt) == 0:
                    debug_print(f"Warning: No face detected in {'prediction' if len(faces_pred) == 0 else 'target'} for sample {batch_idx}")
                            zero_embed = np.zeros(512, dtype=np.float32)
                            emb_preds.append(zero_embed)
                            emb_tgts.append(zero_embed)
                            continue
                        
                        # Get the first face embedding
                        emb_preds.append(faces_pred[0].normed_embedding)
                        emb_tgts.append(faces_tgt[0].normed_embedding)
                        
                    except Exception as e:
                        print(f"Error processing face {batch_idx}: {str(e)}")
                        # Use zero embeddings
                        zero_embed = np.zeros(512, dtype=np.float32)
                        emb_preds.append(zero_embed)
                        emb_tgts.append(zero_embed)
                
        # Calculate ID loss
                if emb_preds and emb_tgts:
                    emb_preds_tensor = torch.tensor(np.stack(emb_preds), dtype=torch.float32, device=device)
                    emb_tgts_tensor = torch.tensor(np.stack(emb_tgts), dtype=torch.float32, device=device)
                    
                    # Calculate cosine similarity for each pair
                    cos_sim = torch.sum(emb_preds_tensor * emb_tgts_tensor, dim=1)  # dot product of normalized vectors
                    loss_id = (1.0 - cos_sim).mean()  # average over batch
                    
                    print(f"Identity loss: {loss_id.item()}")
                    print(f"ID loss details: mean_cos_sim={cos_sim.mean().item():.6f}, loss={loss_id.item():.6f}, weighted={lambda_id * loss_id.item():.6f}")
                    
                    # Add to generator loss
                    gen_loss += lambda_id * loss_id
                else:
                    loss_id = torch.tensor(0.0, device=device)
                
            except Exception as e:
                print(f"Error in identity loss calculation: {e}")
                loss_id = torch.tensor(0.0, device=device)
"""

def capture_final_latents(pipe, mesh_imgs, prompt, num_steps, guidance_scale):
    """
    Runs Img2Img with a callback and returns the final latent tensor [B,C,H,W].
    Must be called *after* pipe.__call__ has been unwrapped.
    """
    latents_list = []
    def _capture(pipeline, step, timestep, callback_kwargs):
        if step == 5: # TODO: remove hardcoded step
            latents_list.append(callback_kwargs["latents"].clone())
        return callback_kwargs

    # run the pipeline under full autograd
    out = pipe(
        image=mesh_imgs,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        callback_on_step_end=_capture,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    
    return latents_list[0], out

def denoise_for_gan(
    pipe: StableDiffusionXLImg2ImgPipeline,
    init_image: torch.Tensor,     # [B,3,H,W] in [-1,1]
    prompt: str,
    num_inference_steps: int,
    strength: float,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Runs SDXL img2img exactly as pipe() would, but under autograd.
    Returns final latents [B, C, H, W] with requires_grad=True.
    """

    # 1) Scheduler: same class & config as the pipeline uses
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 2) Get timesteps & sigmas via the pipeline's own helper
    timesteps, _ = pipe._get_timesteps(
        num_inference_steps=num_inference_steps,
        strength=strength,
    )    

    # 3) Prepare latents exactly as the pipeline does (VAE encode + scaling + noise)
    init_image = pil_to_tensor(init_image, device=device)
    latents = pipe.prepare_latents(
        image=init_image,               # your [B,3,H,W] tensor
        timestep=int(timesteps[0].item()),                # int scalar from get_timesteps()[0]
        batch_size=init_image.shape[0],                   # e.g. init_image.shape[0] × num_per_prompt
        num_images_per_prompt=1,        # usually 1 in GAN-training
        dtype=scheduler.config.sample_dtype,          # match your U-Net dtype
        device=device
    ).requires_grad_(True)

    # 4) Encode the prompt once for classifier-free guidance
    bsz = init_image.shape[0]
    cond, uncond, cond_p, uncond_p = pipe.encode_prompt(
        [prompt]*bsz,
        device=device,
        do_classifier_free_guidance=True,
    )

    # 5) Reverse diffusion loop under autograd
    for i, t in enumerate(timesteps.tolist()):
        # a) Prepare model input for CFG
        lat_in = torch.cat([latents, latents], dim=0)

        # b) Scale for EulerDiscreteScheduler
        lat_scaled = scheduler.scale_model_input(lat_in, t)

        # c) UNet noise prediction
        noise_pred = pipe.unet(
            lat_scaled, 
            t,
            encoder_hidden_states=torch.cat([uncond, cond], dim=0),
            added_cond_kwargs={
                "orig_image_latents": lat_in,          # mesh or original latents
                "text_embeds": torch.cat([uncond_p, cond_p], dim=0),
                "time_ids": None,       # uses default in SDXL pipeline
                "neg_time_ids": None,
            },
        ).sample

        # d) CFG mixing
        uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
        eps = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        # e) One denoising step
        latents = scheduler.step(eps, t, latents).prev_sample

    return latents

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
    aura_model,
    disc,
    seq_embeds,
    pooled_embeds,
    vgg,
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
            seq_embeds, pooled_embeds, device
        )
        
        # Calculate MSE and patch losses
        mse_loss, patch_loss = calculate_mse_patch_loss(noise_pred, noise, config)
        
        # Initialize total loss
        total_loss = (
            config["loss_spatial"]["lambda_mse"] * mse_loss +
            config["loss_spatial"]["lambda_patch"] * patch_loss
        )
        
        # Get denoised latents for GAN if needed
        fake_latents = None
        if is_training and epoch >= config["schedule"]["spatial"]["gan_start"] and step_idx % config["loss_spatial"]["gan_frequency"] == 0:
            # fake_latents = denoise_for_gan(
            #     pipe,
            #     init_image=mesh_imgs,
            #     prompt=config["diffusion"]["prompt"],
            #     num_inference_steps=config["diffusion"]["num_timesteps"],
            #     strength=config["diffusion"]["denoising_strength"],
            #     guidance_scale=config["diffusion"]["guidance_scale"],
            #     device=device,
            # )
            #debug_print("denoise_for_gan done")
            fake_latents, out = capture_final_latents(
                pipe,
                mesh_imgs,
                prompt=config["diffusion"]["prompt"],
                num_inference_steps=config["diffusion"]["num_timesteps"],
                guidance_scale=config["diffusion"]["guidance_scale"]
            )
            print("requires_grad on latents:", fake_latents.requires_grad)  # -> True
            
            # Convert both real and fake latents to tensors in [-1,1] range
            fake_imgs = []
            real_imgs = []
            
            # Process all images in batch for discriminator
            for i in range(fake_latents.shape[0]):
                # Process fake image
                fake_tensor = latent_to_image(pipe, fake_latents, device, i, return_type="tensor")
                fake_imgs.append(fake_tensor.unsqueeze(0))  # [1,3,H,W]
                
                # Process real image
                real_tensor = latent_to_image(pipe, ref_latents, device, i, return_type="tensor")
                real_imgs.append(real_tensor.unsqueeze(0))  # [1,3,H,W]
            
            # Stack tensors
            fake = torch.cat(fake_imgs, dim=0).to(device)
            real = torch.cat(real_imgs, dim=0).to(device)
            
            # Save debug preview if output_dir is provided
            if output_dir is not None and step_idx % 2 == 0:
                debug_print("Saving debug preview")
                preview_dir = Path(output_dir) / "latent_debug"
                preview_dir.mkdir(exist_ok=True, parents=True)
                preview_path = preview_dir / f"latent_preview_{step_idx}.png"
                
                # Create a grid showing both real and fake images
                W, H = 512, 512  # Assuming 512x512 images
                grid = Image.new('RGB', (W * 2, H))
                
                fake_pil = latent_to_image(pipe, fake_latents, device, 0, return_type="pil")
                real_pil = latent_to_image(pipe, ref_latents, device, 0, return_type="pil")
                
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
        
        # Add perceptual loss if applicable
        perc_loss = torch.tensor(0.0, device=device)
        if epoch >= config["schedule"]["spatial"]["perceptual_start"] and step_idx % config["loss_spatial"]["lpips_frequency"] == 0:
            if fake_latents is None: # single-step if GAN isn't active
                fake_latents = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timesteps,
                    sample=noisy_ref
                ).prev_sample
                print("single step done")
            perc_loss = calculate_perceptual_loss(pipe, fake_latents, ref_latents, vgg, lpips_loss_fn, device)
            total_loss += config["loss_spatial"]["lambda_lpips"] * perc_loss
    
    # Training-specific updates
    if is_training:
        # Discriminator update if applicable
        if epoch >= config["schedule"]["spatial"]["gan_start"] and step_idx % config["loss_spatial"]["gan_frequency"] == 0:
            debug_print("\n[Discriminator Update]")
            optimizer_D.zero_grad(set_to_none=True)
            
            # Discriminator forward pass
            logits_real = disc(real)
            logits_fake = disc(fake)
            
            # Discriminator loss
            loss_D_real = F.relu(1.0 - logits_real).mean(dim=None)
            loss_D_fake = F.relu(1.0 + logits_fake).mean(dim=None)
            loss_D = loss_D_real + loss_D_fake
            
            # Update discriminator
            accelerator.backward(loss_D)
            optimizer_D.step()
            optimizer_D.zero_grad()
            
            losses.update({
                "loss_D": loss_D.item(),
                "loss_D_real": loss_D_real.item(),
                "loss_D_fake": loss_D_fake.item(),
            })
            
            # Add GAN loss to generator
            logits_fake_for_G = disc(fake)
            loss_G_adv = -logits_fake_for_G.mean(dim=None)
            total_loss += config["loss_spatial"]["lambda_adv"] * loss_G_adv
            losses["loss_G_adv"] = loss_G_adv.item()
        else:
            losses.update({
                "loss_D": 0.0,
                "loss_D_real": 0.0,
                "loss_D_fake": 0.0,
                "loss_G_adv": 0.0,
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
        "id_loss": 0.0,
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
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=float(config["training"]["learning_rate"]) * 0.5)

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
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        sdxl_cache_path,
        subfolder="scheduler"
    )

    # ─── Stage 7: Setup loss functions and perceptual models ───
    # AuraFace perceptual model
    modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
    aura_root = os.path.join(modal_cache, "insightface", "AuraFace-v1")
    aura_model = FaceAnalysis(
        name="auraface",      # matches the subfolder under models/
        root=aura_root,
        providers=["CUDAExecutionProvider"],  # or CPUExecutionProvider if no GPU
    )
    
    # choose a small det_size since our crops are tight
    aura_model.prepare(ctx_id=0, det_thresh=0.1, det_size=(256, 256))
    debug_print(f">>> det_thresh={aura_model.det_thresh}   det_size={aura_model.det_size}")
    debug_print(">>> aura_model attrs:", [a for a in dir(aura_model) if not a.startswith("_")])

    
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
    optimizer, train_loader, val_loader, lr_scheduler, video_loader, disc, opt_disc, lpips_loss_fn, noise_scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, lr_scheduler, video_loader, disc, opt_disc, lpips_loss_fn, noise_scheduler
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
                            lpips_loss_fn, aura_model, disc,
                            seq_embeds, pooled_embeds,
                            vgg,
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
                    weighted_lpips = config["loss_spatial"]["lambda_lpips"] * losses.get('perc_loss', 0.0)
                    weighted_gan_g = config["loss_spatial"]["lambda_adv"] * losses.get('loss_G_adv', 0.0)
                    
                    accelerator.print(
                        f"[Spatial] Step {global_step} | LR {optimizer.param_groups[0]['lr']:.5f} | "
                        f"Loss {loss:.4f} | "
                        f"MSE {losses['mse_loss']:.4f} (w: {weighted_mse:.4f}) | "
                        f"Patch {losses['patch_loss']:.4f} (w: {weighted_patch:.4f}) | "
                        f"Perceptual {losses.get('perc_loss', 0.0):.4f} (w: {weighted_lpips:.4f}) | "
                        f"GAN G {losses.get('loss_G_adv', 0.0):.4f} (w: {weighted_gan_g:.4f}) | "
                        f"GAN D {losses.get('loss_D', 0.0):.4f}"
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
                        lpips_loss_fn, aura_model, disc,
                        seq_embeds, pooled_embeds,
                        vgg,
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

    # ─── B) TEMPORAL TRAINING ────────────────────────────────────────────
    """
    # Latent-phase
    best_latent, latent_patience = float('inf'), 0
    for t_epoch in range(config["schedule"]["temporal"]["latent_phase_epochs"]):
        pipe.unet.train()
        total_loss = 0.0
        
        # use a small random subset of your video frames each epoch
        few = config["loss_temporal"]["pairs_per_epoch"]
        for i, (mesh_t, mesh_t1, flow) in enumerate(video_loader):
            if i >= few: break
            mesh_t  = mesh_t.to(device)
            mesh_t1 = mesh_t1.to(device)
            flow    = flow.to(device)
            # warp mesh_t → mesh_t1 using your flow
            B,C,H,W = mesh_t.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1,1,H,device=device),
                torch.linspace(-1,1,W,device=device),
                indexing='ij'
            )
            grid = torch.stack((xx, yy), -1).unsqueeze(0).repeat(B,1,1,1)
            flow_norm = torch.zeros_like(grid, device=device)
            flow_norm[...,0] = flow[:,0]/(W/2)
            flow_norm[...,1] = flow[:,1]/(H/2)
            warped = F.grid_sample(mesh_t.unsqueeze(0), grid+flow_norm, align_corners=True)
            loss_temp = F.l1_loss(warped, mesh_t1.unsqueeze(0))
            accelerator.backward(loss_temp * config["loss_temporal"]["lambda_id"])
            optimizer.step()
            optimizer.zero_grad()

    # Pixel-polish phase
    best_pixel, pixel_patience = float('inf'), 0
    for p_epoch in range(config["schedule"]["temporal"]["pixel_phase_epochs"]):
        train_temporal_pixel_epoch(pipe, optimizer, lr_scheduler,
                                   video_loader, config, lpips_fn, ema)
        val_pixel = validate_temporal_pixel(pipe, video_loader, config, lpips_fn)
        if val_pixel < best_pixel:
            best_pixel, pixel_patience = val_pixel, 0
            save_checkpoint(pipe, ema, f"best_temporal_pixel")
        else:
            pixel_patience += 1
            if pixel_patience >= config["training"]["temporal_early_stop_patience"]:
                print("Early stopping TEMPORAL PIXEL")
                break"""

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