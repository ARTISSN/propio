# scripts/train_lora.py

import os
import re
import json
import torch
import yaml
import modal
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.nn import MSELoss, CosineEmbeddingLoss, functional as F
from peft.tuners.lora import LoraLayer
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from torchvision.utils import save_image
import datetime
import numpy as np
from typing import Optional
from diffusers.training_utils import EMAModel
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DDPMScheduler
)
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from lpips import LPIPS

from swapper.utils.image_utils import preprocess_image
from swapper.utils.embedding_utils import get_face_embedding
from swapper.utils.training_modules import JointModel, AdaptiveLoraLayer

# Debug configuration
DEBUG_MODE = False  # Set to True to enable detailed debugging output

def debug_print(*args, **kwargs):
    """Wrapper for print that only outputs if DEBUG_MODE is True"""
    if DEBUG_MODE:
        print(*args, **kwargs)

# Define mount paths
MOUNT_ROOT = "/workspace"
DATA_MOUNT = f"{MOUNT_ROOT}/data/characters"
CACHE_MOUNT = f"{MOUNT_ROOT}/cache"  # Single mount point for all cached/persistent data

# Define volumes for persistent storage
CHARACTER_DATA_VOLUME = modal.Volume.from_name("character-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)  # Single volume for all caches

def encode_image(vae, image):
    """Encode image to latent space."""
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents

# helper to locate parent and attribute name from a dotted path
def _find_parent_module(root, full_name):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

# --------------- Dataset ----------------
# --- Dataset ---
class CharacterDataset(Dataset):
    def __init__(self, character_dir, config):
        print(f"\n=== Initializing CharacterDataset ===")
        print(f"Character directory: {character_dir}")
        
        self.config = config
        character_dir = Path(character_dir)  # Convert to Path object
        self.faces_dir = character_dir / "processed/maps/faces"
        self.normals_dir = character_dir / "processed/maps/normals"
        self.meta_path = character_dir / "metadata.json"
        
        print(f"Checking paths:")
        print(f"- Faces dir exists: {self.faces_dir.exists()}")
        print(f"- Normals dir exists: {self.normals_dir.exists()}")
        print(f"- Metadata exists: {self.meta_path.exists()}")

        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)["frames"]

        # Add this debug section
        debug_print("\nDebugging metadata structure:")
        debug_print(f"Number of frames in metadata: {len(self.metadata)}")
        sample_frame = next(iter(self.metadata.items()))
        debug_print(f"\nSample frame structure:")
        debug_print(f"Frame ID: {sample_frame[0]}")
        debug_print(f"Frame data keys: {sample_frame[1].keys()}")
        if "embedding" in sample_frame[1]:
            embedding = sample_frame[1]["embedding"]
            debug_print(f"Embedding type: {type(embedding)}")
            debug_print(f"Embedding length: {len(embedding) if isinstance(embedding, (list, tuple)) else 'N/A'}")
        else:
            debug_print("No embedding found in sample frame")
        
        print(f"\nProcessing samples...")
        self.samples = []
        for frame_id, frame_data in self.metadata.items():
            if isinstance(frame_data, dict) and "maps" in frame_data:
                # Fix path handling: remove extra 'characters' directory and normalize slashes
                face_path_str = frame_data["maps"]["face"].replace("\\", "/")
                normal_path_str = frame_data["maps"]["normal"].replace("\\", "/")
                
                # Remove duplicate 'characters/character_name' if present
                face_path_str = face_path_str.replace(f"characters/{character_dir.name}/", "", 1)
                normal_path_str = normal_path_str.replace(f"characters/{character_dir.name}/", "", 1)
                
                # Construct absolute paths
                face_path = character_dir / face_path_str
                normal_path = character_dir / normal_path_str
                
                print(f"Checking paths for {frame_id}:")
                print(f"- Face path: {face_path}")
                print(f"- Normal path: {normal_path}")
                
                # Verify files exist
                if not face_path.exists():
                    print(f"Warning: Face map not found: {face_path}")
                    continue
                if not normal_path.exists():
                    print(f"Warning: Normal map not found: {normal_path}")
                    continue
                
                sample = {
                    "image": str(face_path),
                    "normal_map": str(normal_path),
                    "lighting": frame_data.get("lighting", {}).get("coefficients"),
                    "embedding": embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding,
                    "frame_id": frame_id,
                }
                if all(v is not None for v in sample.values()):
                    self.samples.append(sample)
                    print(f"Added sample: {frame_id}")
                    print(f"- Face path: {sample['image']}")
                    print(f"- Normal path: {sample['normal_map']}")
        
        print(f"Found {len(self.samples)} valid samples")

        # Check embedding dimension from first valid sample
        if self.samples:
            self.embedding_dim = len(self.samples[0]["embedding"])
            print(f"Found embedding dimension: {self.embedding_dim}")
        else:
            self.embedding_dim = 128  # default
            print(f"No samples found, using default embedding dimension: {self.embedding_dim}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Add error handling for image loading
        try:
            image = preprocess_image(sample["image"], target_size=(self.config["resolution"], self.config["resolution"]))
            normal = preprocess_image(sample["normal_map"], target_size=(self.config["controlnet_resolution"], self.config["controlnet_resolution"]))
        except Exception as e:
            print(f"Error loading images for sample {sample['frame_id']}:")
            print(f"- Face path: {sample['image']}")
            print(f"- Normal path: {sample['normal_map']}")
            print(f"Error: {str(e)}")
            raise
        
        # Convert image to bfloat16 to match model dtype
        pixel_values = torch.tensor(image, dtype=torch.bfloat16).permute(2, 0, 1)
        normal_tensor = torch.from_numpy(normal).to(dtype=torch.bfloat16).permute(2, 0, 1)

        # Use stored embedding from metadata
        embedding = torch.tensor(sample["embedding"], dtype=torch.float32)
        lighting = torch.tensor(sample["lighting"], dtype=torch.float32)

        # Before feeding into ControlNet
        normal_tensor = normal_tensor * 2.0 - 1.0  # Scale from [0,1] to [-1,1]

        return {
            "pixel_values": pixel_values,
            "normal_map": normal_tensor,
            "embedding": embedding,
            "lighting": lighting,
        }

    def __len__(self):
        return len(self.samples)

# --------------- Helper functions ----------------

def prepare_latents(vae, image_tensor, scheduler):
    with torch.no_grad():
        # Ensure image_tensor is in the same dtype as the VAE
        image_tensor = image_tensor.to(dtype=vae.dtype)
        latents = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, noise, timesteps

# --------------- Training ----------------
def load_checkpoint(checkpoint_path: Path, pipe, optimizer, lr_scheduler, unet_ema, controlnet_ema, fusion_ema):
    """Load model and training state from checkpoint."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    try:
        # Load model weights
        unet_path = checkpoint_path / "unet_lora.pt"
        controlnet_path = checkpoint_path / "controlnet_lora.pt"
        training_state_path = checkpoint_path / "training_state.pt"
        
        required_files = [
            unet_path, controlnet_path, training_state_path
        ]
        
        if not all(p.exists() for p in required_files):
            raise FileNotFoundError("Checkpoint files are incomplete")
            
        # Load model weights
        pipe.unet.load_state_dict(torch.load(unet_path))
        pipe.controlnet.load_state_dict(torch.load(controlnet_path))
        
        # Load training state
        training_state = torch.load(training_state_path)
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(training_state["optimizer"])
        lr_scheduler.load_state_dict(training_state["scheduler"])
        
        # Load EMA states
        if "ema" in training_state:
            unet_ema.load_state_dict(training_state["ema"]["unet"])
            controlnet_ema.load_state_dict(training_state["ema"]["controlnet"])
            fusion_ema.load_state_dict(training_state["ema"]["fusion"])
        
        return {
            "step": training_state["step"],
            "epoch": training_state["epoch"],
            "config": training_state["config"]
        }
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def save_checkpoint(checkpoint_path: Path, pipe, optimizer, lr_scheduler, unet_ema, controlnet_ema, fusion_ema, step, epoch, config):
    """Save model and training state to checkpoint."""
    try:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(pipe.unet.state_dict(), checkpoint_path / "unet_lora.pt")
        torch.save(pipe.controlnet.state_dict(), checkpoint_path / "controlnet_lora.pt")
        
        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "config": config,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "ema": {
                "unet": unet_ema.state_dict(),
                "controlnet": controlnet_ema.state_dict(),
                "fusion": fusion_ema.state_dict()
            }
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        print("Checkpoint saved successfully")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Add this function definition before the preview generation code block
def validate_pipeline_inputs(validation_inputs):
    """Validate pipeline inputs before processing."""
    # 1. Check prompt/prompt_embeds mutual exclusivity
    if validation_inputs["prompt"] is not None and validation_inputs["prompt_embeds"] is not None:
        raise ValueError("Provide either `prompt` or `prompt_embeds`, not both")
    
    # 2. Check that at least one of prompt or prompt_embeds is provided
    if validation_inputs["prompt"] is None and validation_inputs["prompt_embeds"] is None:
        raise ValueError("Either `prompt` or `prompt_embeds` must be provided")
    
    # 3. Check image
    if validation_inputs["image"] is None:
        raise ValueError("`image` input cannot be None")
    
    if not isinstance(validation_inputs["image"], torch.Tensor):
        raise ValueError(f"`image` must be a torch.Tensor, got {type(validation_inputs['image'])}")
    
    # 4. Check tensor shapes
    if validation_inputs["prompt_embeds"] is not None:
        if validation_inputs["prompt_embeds"].ndim != 3:
            raise ValueError(f"prompt_embeds must be 3-dimensional, got {validation_inputs['prompt_embeds'].ndim} dimensions")
        if validation_inputs["prompt_embeds"].shape[-1] != 2048:
            raise ValueError(f"prompt_embeds must have shape [..., 2048], got [..., {validation_inputs['prompt_embeds'].shape[-1]}]")

    # 5. Check device consistency
    if validation_inputs["prompt_embeds"] is not None and validation_inputs["pooled_prompt_embeds"] is not None:
        if validation_inputs["prompt_embeds"].device != validation_inputs["pooled_prompt_embeds"].device:
            raise ValueError("prompt_embeds and pooled_prompt_embeds must be on the same device")

    # 6. Check dtype consistency
    if validation_inputs["prompt_embeds"] is not None:
        if validation_inputs["prompt_embeds"].dtype != torch.bfloat16:
            raise ValueError(f"prompt_embeds must be of dtype bfloat16, got {validation_inputs['prompt_embeds'].dtype}")

    return True

def validate_projector_outputs(combined_cross, controlnet_embeds, pooled_embeds, pipe):
    """Validate the dimensions of projector outputs."""
    assert combined_cross.shape[-1] == pipe.unet.config.cross_attention_dim, \
        f"UNet cross attention dimension mismatch: {combined_cross.shape[-1]} vs {pipe.unet.config.cross_attention_dim}"
    
    assert controlnet_embeds.shape[-1] == pipe.controlnet.config.cross_attention_dim, \
        f"ControlNet cross attention dimension mismatch: {controlnet_embeds.shape[-1]} vs {pipe.controlnet.config.cross_attention_dim}"
    
    assert pooled_embeds.shape[-1] == pipe.text_encoder_2.config.projection_dim, \
        f"Pooled embeddings dimension mismatch: {pooled_embeds.shape[-1]} vs {pipe.text_encoder_2.config.projection_dim}"

def validate_conditioning_inputs(pooled_embeds, batch_size, device):
    """Validate the conditioning inputs for SDXL."""
    assert pooled_embeds.ndim == 2, f"Pooled embeddings should be 2D, got {pooled_embeds.ndim}D"
    assert pooled_embeds.shape[-1] == 1280, f"Pooled embeddings should be 1280-dimensional, got {pooled_embeds.shape[-1]}"
    assert pooled_embeds.shape[0] == batch_size, f"Batch size mismatch: {pooled_embeds.shape[0]} vs {batch_size}"
    assert pooled_embeds.device == device, f"Device mismatch: {pooled_embeds.device} vs {device}"
    return True

def calculate_losses(
    noise_pred, noise, noise_scheduler, timesteps, noisy_latents, real_images, vae, 
    global_step=0, warmup_steps=1000, id_loss_weight=0.5, calc_id_every=5, lora_params=None,
    shape_predictor_path=None, face_rec_model_path=None, debug_dir: Optional[Path] = None
):
    # Basic diffusion loss
    diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
    
    # Only do identity loss if weight is positive and it's after warmup
    identity_weight = min(1.0, global_step / max(1, warmup_steps)) * id_loss_weight
    identity_loss = torch.tensor(0.0, device=diffusion_loss.device)
    
    # Skip identity loss calculation if weight is 0 or too early in training
    if identity_weight > 0 and global_step > warmup_steps // 2 and (global_step % calc_id_every == 0):
        try:
            # Move scheduler to same device as tensors
            noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
            
            # Process batch in smaller chunks if needed
            batch_size = timesteps.shape[0]
            
            # We need to handle each timestep separately since they may differ in the batch
            prev_samples = []
            for i in range(batch_size):
                # Process one sample at a time
                sample_timestep = timesteps[i:i+1]  # Keep dim for broadcasting
                sample_noise_pred = noise_pred[i:i+1]
                sample_latents = noisy_latents[i:i+1]
                
                # Call step for this individual sample
                sample_prev = noise_scheduler.step(
                    sample_noise_pred, 
                    sample_timestep, 
                    sample_latents
                ).prev_sample
                
                prev_samples.append(sample_prev)
            
            # Stack back into a batch
            prev = torch.cat(prev_samples, dim=0)
            
            # *** CRITICAL FIX - Ensure correct dtype for VAE decoding ***
            # 3) decode to RGB via the VAE - maintain the correct dtype
            latents = prev / vae.config.scaling_factor  
            latents = latents.to(dtype=vae.dtype)  # Match VAE's dtype (bfloat16)
            
            with torch.no_grad():
                decoded = vae.decode(latents).sample            # [B, C, H, W]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)       # to [0,1]
            
            # 4) get the real face in [0,1]
            real = (real_images / 2 + 0.5).clamp(0, 1)      # [B, C, H, W]
            
            # Add additional checks before face embedding
            if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                print("WARNING: NaN or Inf values detected in decoded images. Skipping identity loss.")
                raise ValueError("Invalid decoded values")
                
            # Check if the images have proper content before face embedding
            if decoded.min() == decoded.max():
                print("WARNING: Decoded images have no variation. Skipping identity loss.")
                raise ValueError("No image variation")
            
            # Move to CPU for face detection processing
            decoded_np = decoded.cpu().float().numpy()
            real_np = real.cpu().float().numpy()
            
            # Convert to the expected format for face detection (uint8, 0-255)
            decoded_np = (decoded_np * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            real_np = (real_np * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            
            # Additional check for face detector path
            if shape_predictor_path is None or not os.path.exists(shape_predictor_path):
                print(f"WARNING: Face detector model not found at {shape_predictor_path}. Skipping identity loss.")
                raise ValueError("Face detector not found")
            else:
                print(f"Face detector model found at {shape_predictor_path}")

            # Process each image individually
            for i in range(decoded_np.shape[0]):
                single_decoded_np = decoded_np[i]
                single_real_np = real_np[i]

                # Call face embedding for each image
                try:
                    pred_emb = get_face_embedding(single_decoded_np, shape_predictor_path, face_rec_model_path, debug_dir)
                    true_emb = get_face_embedding(single_real_np, shape_predictor_path, face_rec_model_path, debug_dir)
                    
                    if pred_emb is None or true_emb is None:
                        raise ValueError("Face embeddings returned None")
                    
                    target = torch.ones(pred_emb.size(0), device=pred_emb.device)
                    id_loss = F.cosine_embedding_loss(pred_emb.float(), true_emb.float(), target)
                    identity_loss += id_loss  # Accumulate identity loss
                except Exception as e:
                    print(f"WARNING: Face embedding error for image {i}: {str(e)}. Skipping identity loss for this image.")
                    continue

            # Average identity loss over the batch
            identity_loss /= decoded_np.shape[0]

        except Exception as e:
            print(f"WARNING: Error in identity loss calculation: {str(e)}")
            print("Skipping identity loss for this batch")
            identity_loss = torch.tensor(0.0, device=diffusion_loss.device)
            torch.cuda.empty_cache()

    if lora_params is not None:
        reg = 1e-6 * sum(p.pow(2).sum() for p in lora_params.values())
    else:
        reg = 0.0
    
    # Total loss
    total = diffusion_loss + identity_weight * identity_loss + reg
    
    return {
        "total_loss": total,
        "diffusion_loss": diffusion_loss,
        "identity_loss": identity_loss,
        "identity_weight": identity_weight,
    }

def prepare_latents_and_conditioning(
    batch, pipe, fusion, unet_proj, ctrl_proj, config, device
):
    # Extract inputs from the batch
    image = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    normal_map = batch["normal_map"].to(device, dtype=torch.bfloat16)
    identity = F.normalize(batch["embedding"].to(device), dim=-1)
    lighting = F.normalize(batch["lighting"].to(device), dim=-1)

    # Fuse modalities
    mm_embed = fusion(identity, lighting, normal_map)  # [B, hidden_dim]

    # Project to SDXL dims + add sequence dim
    cross_unet = unet_proj(mm_embed).unsqueeze(1)  # [B,1,2048]
    cross_ctrl = ctrl_proj(mm_embed).unsqueeze(1)  # [B,1,2048]

    # Prepare latents
    noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, pipe.scheduler)

    # Create consistent conditioning
    batch_size = noisy_latents.shape[0]

    # Spatial crop/time IDs
    time_ids = torch.zeros((batch_size, 6), device=device)
    time_ids[:, 0] = config["resolution"]  # H
    time_ids[:, 1] = config["resolution"]  # W
    time_ids[:, 2:6] = torch.tensor(
        [0, 0, config["resolution"], config["resolution"]],
        device=device
    )

    # Empty text embeddings
    proj_dim = pipe.text_encoder_2.config.projection_dim
    text_embeds = torch.zeros((batch_size, proj_dim), dtype=torch.bfloat16, device=device)

    added_cond_kwargs = {
        "time_ids": time_ids,  # spatial region for ControlNet
        "text_embeds": text_embeds  # no-text-conditioning placeholder
    }

    return noisy_latents, noise, timesteps, cross_unet, cross_ctrl, added_cond_kwargs, normal_map

def replace_lora_with_adaptive(model, cond_dim):
    """Replace LoraLayer with AdaptiveLoraLayer in the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            parent, attr = _find_parent_module(model, name)
            setattr(parent, attr, AdaptiveLoraLayer(module, cond_dim))

def train(character_name: str, output_dir: Optional[str] = None, from_checkpoint: bool = False):
    """Train the model with support for checkpoints and multiple runs."""
    try:
        global DLIB_SHAPE_PREDICTOR_PATH
        print("\n=== Starting training setup ===")
        
        # Use provided output directory or create default
        output_dir = Path(output_dir) if output_dir else Path(DATA_MOUNT) / character_name / "trainings" / f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\nTraining output will be saved to: {output_dir}")
        
        # Initialize checkpoint variables
        start_epoch = 0
        global_step = 0
        checkpoint_state = None
        
        # Load checkpoint if specified
        if from_checkpoint:
            print("\nLooking for checkpoint...")
            checkpoint_dir = output_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    print(f"Found latest checkpoint: {latest_checkpoint}")
                else:
                    print("No checkpoints found, starting fresh training")
                    from_checkpoint = False
            else:
                print("No checkpoints directory found, starting fresh training")
                from_checkpoint = False

        # Use absolute paths with Path objects
        data_mount = Path(DATA_MOUNT)  # From your constants
        character_path = data_mount / character_name
        
        print(f"\nChecking character path:")
        print(f"- Data mount: {data_mount}")
        print(f"- Character path: {character_path}")
        print(f"- Character path exists: {character_path.exists()}")
        
        # List contents to verify structure
        if character_path.exists():
            print("\nCharacter directory contents:")
            for item in character_path.rglob("*"):
                print(f"- {item.relative_to(character_path)}")
        
        # Use the Modal cache path for the model
        modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
        landmarks_path = os.path.join(modal_cache, "models", "shape_predictor_68_face_landmarks.dat")
        
        # 3. Now safe to initialize dataset
        with open("configs/train_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        print("\nLoading dataset...")
        print(f"Character path: {character_path}")
        print(f"Character path exists: {character_path.exists()}")

        try:
            dataset = CharacterDataset(character_path, config)
            print(f"Dataset created successfully with {len(dataset)} samples")
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        print(f"Found {len(dataset)} valid samples (expected 31)")
        for s in dataset.samples:
            print(s["frame_id"], s["image"], s["normal_map"])
            
        accelerator = Accelerator(
            mixed_precision=config["optimization"]["mixed_precision"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
        )

        # Split dataset into train and validation sets
        val_fraction = 0.2  # 20% for validation
        val_size = int(len(dataset) * val_fraction)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

        # Get the cache path from environment
        modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
        sdxl_cache_path = os.path.join(modal_cache, "huggingface", "sdxl-base-1.0")

        print(f"\nLoading SDXL model from cache:")
        print(f"- Cache path: {sdxl_cache_path}")
        print(f"- Path exists: {os.path.exists(sdxl_cache_path)}")

        try:
            base = StableDiffusionXLPipeline.from_pretrained(
                sdxl_cache_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
            ).to(accelerator.device)

            controlnet = ControlNetModel.from_unet(base.unet)
            #controlnet.config.addition_embed_type = None

            pipe = StableDiffusionXLControlNetPipeline(
                vae=base.vae,
                text_encoder=base.text_encoder,
                text_encoder_2=base.text_encoder_2,
                tokenizer=base.tokenizer,
                tokenizer_2=base.tokenizer_2,
                image_encoder=base.image_encoder,
                feature_extractor=base.feature_extractor,
                unet=base.unet,
                controlnet=controlnet,
                scheduler=base.scheduler,
            ).to(accelerator.device)

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # First, create the LoRA config with the correct module names
        try:
            # instead of EmbeddingProjector, build:
            fuse_dim = config["model"]["hidden_dim"]  # e.g. 512
            spatial_dim = 3  # your normal_map channels
            id_dim      = dataset.embedding_dim       # e.g. 128
            light_dim   = config["model"]["lighting_dim"]  # e.g. 16
            unet_dim    = config["model"]["output_dim"]
            ctrl_dim    = config["model"]["output_dim"]

            joint_model = JointModel(id_dim, light_dim, spatial_dim, fuse_dim, hidden_dim, unet_dim, ctrl_dim).to(accelerator.device)
            
            fusion_lora_config = LoraConfig(
                r=config["model"]["lora_rank"],
                lora_alpha=config["model"]["lora_alpha"],
                lora_dropout=config["model"]["lora_dropout"],
                target_modules=[
                    # the three projection layers:
                    "id_proj",         # projects identity
                    "light_proj",      # projects lighting
                    "spatial_proj",    # projects normal_map
                    # plus the internal self-attention:
                    "attn.in_proj_weight",
                    "attn.out_proj.weight",
                ],
                bias="none",
                task_type="CUSTOM"   # not UNET/CONTROLNET
            )

            identity_lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    # UNet cross-attention projections
                    "unet.down_blocks.*.attentions.*.to_q",
                    "unet.down_blocks.*.attentions.*.to_k",
                    "unet.down_blocks.*.attentions.*.to_v",
                    "unet.mid_block.attentions.*.to_q",
                    "unet.mid_block.attentions.*.to_k",
                    "unet.mid_block.attentions.*.to_v",
                    "to_out.0",
                ],
                bias="none",
                task_type="UNET"
            )

            geometry_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    # ControlNet conditioning & attention
                    "controlnet_cond_embedding.conv_in",
                    "controlnet_cond_embedding.conv_out",
                    "controlnet_down_blocks.*.attentions.*.to_q",
                    "controlnet_down_blocks.*.attentions.*.to_k",
                    "controlnet_down_blocks.*.attentions.*.to_v",
                ],
                bias="none",
                task_type="CONTROLNET"
            )

            lighting_lora_config = LoraConfig(
                r=4,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    # Your LightingMLP internal layers
                    "lighting_mlp.*",
                ],
                bias="none",
                task_type="CUSTOM"
            )

            # Apply LoRA **only** to your fusion module
            pipe.unet = get_peft_model(pipe.unet, identity_lora_config)
            pipe.controlnet = get_peft_model(pipe.controlnet, geometry_lora_config)
            lighting_mlp = get_peft_model(lighting_mlp, lighting_lora_config)
            fusion = get_peft_model(joint_model, fusion_lora_config)

            replace_lora_with_adaptive(pipe.unet,    fuse_dim)
            replace_lora_with_adaptive(pipe.controlnet, fuse_dim)
            replace_lora_with_adaptive(lighting_mlp, fuse_dim)
            replace_lora_with_adaptive(fusion,       fuse_dim)

            # Freeze UNet, ControlNet, Lighting LoRAs so only fusion LoRA trains
            for p in pipe.unet.parameters():       p.requires_grad = False
            for p in pipe.controlnet.parameters(): p.requires_grad = False
            for p in lighting_mlp.parameters():    p.requires_grad = False

            # But keep fusion's LoRA-params trainable
            for name, param in fusion.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            
        except Exception as e:
            print(f"Error configuring LoRA: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # 4. Verify—it should have stamped AdaptiveLoraLayer in place of every LoRA adapter:
        for name, module in fusion.named_modules():
            if isinstance(module, AdaptiveLoraLayer):
                print(f"✅ Adaptively wrapped: {name}")

        # And print their names:
        for name, p in fusion.named_parameters():
            if p.requires_grad:
                print("  🟢", name)

        # Add these debug statements after loading the SDXL pipeline
        debug_print("\n=== Debugging Pipeline Components ===")
        debug_print(f"Pipeline components loaded: {pipe is not None}")
        debug_print(f"Text Encoder 2 exists: {hasattr(pipe, 'text_encoder_2')}")
        if hasattr(pipe, 'text_encoder_2'):
            debug_print(f"Text Encoder 2 config exists: {hasattr(pipe.text_encoder_2, 'config')}")
            if hasattr(pipe.text_encoder_2, 'config'):
                debug_print(f"Text Encoder 2 config: {pipe.text_encoder_2.config}")
                debug_print(f"Projection dim: {getattr(pipe.text_encoder_2.config, 'projection_dim', None)}")

        # Debug config
        debug_print("\n=== Debugging Config ===")
        debug_print(f"Config type: {type(config)}")
        debug_print(f"Config contents: {json.dumps(config, indent=2)}")
        debug_print(f"Model section exists: {'model' in config}")
        if 'model' in config:
            debug_print(f"Model config: {json.dumps(config['model'], indent=2)}")
            debug_print(f"hidden_dim exists: {'hidden_dim' in config['model']}")
            debug_print(f"lighting_dim exists: {'lighting_dim' in config['model']}")

        try:
            # Ensure config has required keys with more detailed error messages
            if not isinstance(config, dict):
                raise TypeError(f"Config must be a dictionary, got {type(config)}")
            if "model" not in config:
                raise KeyError("Config missing 'model' section")
            if "hidden_dim" not in config["model"]:
                raise KeyError("Config missing 'model.hidden_dim'")
            if "lighting_dim" not in config["model"]:
                raise KeyError("Config missing 'model.lighting_dim'")
            
            # Get projection dimension with error handling
            if not hasattr(pipe, 'text_encoder_2'):
                raise AttributeError("Pipeline missing text_encoder_2")
            if not hasattr(pipe.text_encoder_2, 'config'):
                raise AttributeError("text_encoder_2 missing config")
            
            proj_dim = pipe.text_encoder_2.config.projection_dim
            if proj_dim is None:
                raise ValueError("projection_dim is None in text_encoder_2 config")
            
            print(f"\nSuccessfully retrieved dimensions:")
            print(f"- proj_dim: {proj_dim}")
            print(f"- hidden_dim: {config['model']['hidden_dim']}")
            print(f"- lighting_dim: {config['model']['lighting_dim']}")

            hidden_dim = config["model"]["hidden_dim"]
        except Exception as e:
            print(f"\n❌ Error during initialization:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

        #pipe.vae.disable_gradient_checkpointing()

        #pipe.unet.get_base_model().disable_gradient_checkpointing()

        print("❗ UNet trainable params:", 
            sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad))
        print("❗ ControlNet trainable params:", 
            sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad))

        try:
            # Update optimizer to use only UNet and projector parameters
            # 1. Collect only the trainable parameters (all LoRA adapters)
            params_to_optimize = []
            for module in [pipe.unet, pipe.controlnet, lighting_mlp, fusion]:
                for name, p in module.named_parameters():
                    if p.requires_grad:
                        params_to_optimize.append(p)

            # 2. Instantiate AdamW
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=float(config["training"]["learning_rate"]),        # e.g. 1e-4
                betas=(0.9, 0.999),
                weight_decay=float(config["training"]["weight_decay"])  # e.g. 0.01
            )
        except Exception as e:
            print(f"Error in optimizer setup: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # Add to training setup
        num_training_steps = len(train_loader) * config["training"]["num_epochs"]
        num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])

        # Learning rate scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Noise scheduler for diffusion
        noise_scheduler = DDPMScheduler.from_pretrained(
            sdxl_cache_path,
            subfolder="scheduler"
        )

        # Initialize EMA
        try:
            # —––––––– Initialize EMAs before training loop –––––––—
            decay = 0.9999

            # Track UNet's LoRA parameters
            unet_ema = EMAModel(
                pipe.unet.parameters(),
                decay=decay,
                model_cls=pipe.unet.__class__,
                device=accelerator.device
            )

            # Track ControlNet's LoRA parameters
            controlnet_ema = EMAModel(
                pipe.controlnet.parameters(),
                decay=decay,
                model_cls=pipe.controlnet.__class__,
                device=accelerator.device
            )

            # (Optional) Track Fusion's LoRA parameters too
            fusion_ema = EMAModel(
                fusion.parameters(),
                decay=decay,
                model_cls=fusion.__class__,
                device=accelerator.device
            )

            # Then update both EMAs in the training loop
            unet_ema.step(pipe.unet.parameters())
            controlnet_ema.step(pipe.controlnet.parameters())
        except Exception as e:
            print(f"Error in EMA setup: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # Load checkpoint state if available
        if from_checkpoint and latest_checkpoint:
            checkpoint_state = load_checkpoint(
                latest_checkpoint,
                pipe,
                optimizer,
                lr_scheduler,
                unet_ema,
                controlnet_ema,
                fusion_ema
            )
            
            if checkpoint_state:
                start_epoch = checkpoint_state["epoch"] + 1
                global_step = checkpoint_state["step"] + 1
                print(f"Resuming from epoch {start_epoch}, step {global_step}")
                
                # Verify config compatibility
                if checkpoint_state["config"] != config:
                    print("\nWarning: Checkpoint config differs from current config!")
                    print("Using checkpoint config for consistency")
                    config = checkpoint_state["config"]
            else:
                print("Failed to load checkpoint, starting fresh training")
                from_checkpoint = False

        # projection layers
        unet_proj    = nn.Linear(hidden_dim, pipe.unet.config.cross_attention_dim).to(accelerator.device)
        ctrl_proj    = nn.Linear(hidden_dim, pipe.controlnet.config.cross_attention_dim).to(accelerator.device)

        # Prepare models AFTER loading checkpoint
        pipe.unet, pipe.vae, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            pipe.unet, pipe.vae, optimizer, train_loader, val_loader, lr_scheduler
        )

        # Set the paths for the face detection models
        shape_predictor_path = os.path.abspath(landmarks_path)
        face_rec_model_path = os.path.join(modal_cache, "models", "dlib_face_recognition_resnet_model_v1.dat")

        # Define the debug directory path
        debug_dir = Path(output_dir) / "debug_images"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        best_loss = float('inf')
        patience = config["training"].get("patience", 5)
        patience_counter = 0

        for epoch in range(start_epoch, config["training"]["num_epochs"]):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                with accelerator.accumulate():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Inside the training loop, add this before the ControlNet forward pass
                        try:
                            noisy_latents, noise, timesteps, cross_unet, cross_ctrl, added_cond_kwargs, normal_map = prepare_latents_and_conditioning(
                                batch, pipe, fusion, unet_proj, ctrl_proj, config, accelerator.device
                            )

                            # Down blocks and mid blocks residual connections
                            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=cross_ctrl,
                                controlnet_cond=normal_map,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )
                        except Exception as e:
                            debug_print(f"Error in ControlNet forward tracking: {str(e)}")
                            import traceback
                            debug_print(traceback.format_exc())

                        # After backward pass, add this
                        try:
                            debug_print("\n=== Post-Backward Gradient Check ===")
                            missing_grads = []
                            zero_grads = []
                            for name, param in pipe.controlnet.named_parameters():
                                if param.requires_grad:
                                        if param.grad is None:
                                            missing_grads.append(name)
                                        elif param.grad.norm().item() == 0:
                                            zero_grads.append(name)
                                        else:
                                            debug_print(f"Gradient present for {name}: {param.grad.norm().item():.6f}")

                            if missing_grads:
                                debug_print(f"\nWARNING: {len(missing_grads)} parameters have missing gradients:")
                                for name in missing_grads[:10]:  # Print first 10
                                    debug_print(f"- {name}")
                                if len(missing_grads) > 10:
                                    debug_print(f"... and {len(missing_grads) - 10} more")

                            if zero_grads:
                                debug_print(f"\nWARNING: {len(zero_grads)} parameters have zero gradients:")
                                for name in zero_grads[:10]:  # Print first 10
                                    debug_print(f"- {name}")
                                if len(zero_grads) > 10:
                                    debug_print(f"... and {len(zero_grads) - 10} more")
                        except Exception as e:
                            debug_print(f"Error in post-backward gradient check: {str(e)}")

                        try:
                            # UNet forward pass
                            noise_pred = pipe.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=cross_unet,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs
                            ).sample
                        except Exception as e:
                            debug_print(f"Error in UNet forward pass: {str(e)}")
                            import traceback
                            debug_print(traceback.format_exc())

                        try:
                            # Calculate simplified losses
                            losses = calculate_losses(
                                noise_pred,
                                noise,
                                noise_scheduler,     # your DDPMScheduler
                                timesteps,           # the sampled timesteps
                                noisy_latents,       # the latents you passed into UNet
                                batch["pixel_values"],  # batch["pixel_values"]
                                pipe.vae,            # the VAE from your pipeline
                                global_step,
                                warmup_steps=num_warmup_steps,
                                id_loss_weight=0.5,
                                calc_id_every=5,
                                lora_params=None,
                                shape_predictor_path=shape_predictor_path,
                                face_rec_model_path=face_rec_model_path,
                                debug_dir=debug_dir  # Pass the debug directory
                            )
                        except Exception as e:
                            debug_print(f"Error in loss calculation: {str(e)}")
                            import traceback
                            debug_print(traceback.format_exc())

                        # Extract total loss for backward
                        total_loss = losses["total_loss"]
                        accelerator.backward(total_loss)

                        # pick one of your lora tensors
                        for n,p in pipe.controlnet.controlnet_cond_embedding.named_parameters():
                            if "lora_A" in n:
                                debug_print(n, "grad norm =", p.grad.norm().item())
                                break
                        lr_scheduler.step()
                        optimizer.step()

                        # Move EMA update outside accumulation context
                        if accelerator.sync_gradients:
                            unet_ema.step(pipe.unet.parameters())
                            controlnet_ema.step(pipe.controlnet.parameters())
                            fusion_ema.step(fusion.parameters())

                        if global_step % config["training"].get("log_steps", 10) == 0:
                            # Get current learning rate
                            current_lr = optimizer.param_groups[0]["lr"]
                            
                            # Calculate gradients statistics
                            grad_norm_unet = 0.0
                            for name, param in pipe.unet.named_parameters():
                                if param.grad is not None:
                                    grad_norm_unet += param.grad.data.norm(2).item() ** 2
                            grad_norm_unet = grad_norm_unet ** 0.5
                            
                            # Log statistics
                            accelerator.print(
                                f"\n=== Training Statistics ===\n"
                                f"Epoch: {epoch}/{config['training']['num_epochs']} | "
                                f"Step: {global_step}/{config['training']['max_train_steps']} | "
                                f"Learning Rate: {current_lr:.6f}\n"
                                f"Losses:\n"
                                f"- Total: {losses['total_loss'].item():.4f}\n"
                                f"- Diffusion: {losses['diffusion_loss'].item():.4f}\n"
                                f"- Identity (w={losses['identity_weight']:.2f}): {losses['identity_loss'].item():.4f}\n"
                            )
                    
                        try:
                            # Save checkpoints
                            if global_step % config["training"].get("save_steps", 500) == 0:    
                                if accelerator.is_main_process:
                                    save_checkpoint(
                                        checkpoint_path=output_dir / "checkpoints" / f"checkpoint-{epoch}-{global_step}",
                                        pipe=pipe,
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler,
                                        unet_ema=unet_ema, 
                                    )

                                    CHARACTER_DATA_VOLUME.commit()
                                    print("✅ Committed volume changes")
                        except Exception as e:
                            print(f"Error in checkpoint saving: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                            raise

                        # Add the preview generation code:
                        if global_step % config["training"].get("preview_steps", 100) == 0:
                            accelerator.print("\nGenerating preview image...")
                            
                            try:
                                # Add explicit memory cleanup before preview generation
                                torch.cuda.empty_cache()
                                
                                # 1) Put everything in eval mode
                                unet_ema.store(pipe.unet.parameters())
                                controlnet_ema.store(pipe.controlnet.parameters())
                                fusion_ema.store(fusion.parameters())

                                unet_ema.copy_to(pipe.unet.parameters())
                                controlnet_ema.copy_to(pipe.controlnet.parameters())
                                fusion_ema.copy_to(fusion.parameters())

                                pipe.unet.eval()
                                pipe.controlnet.eval()
                                fusion.eval()
                                pipe.vae.eval()

                                device = accelerator.device
                                sample_normal = batch["normal_map"][0:1].to(device)
                                #sample_normal = ((sample_normal + 1.0) / 2.0).clamp(0, 1).to(torch.float32)
                                
                                # Resize if necessary to save memory
                                preview_resolution = min(config["resolution"], 512)
                                if sample_normal.shape[-1] > preview_resolution:
                                    sample_normal = F.interpolate(sample_normal, size=(preview_resolution, preview_resolution), mode='bilinear')
                                
                                # Setup basic parameters
                                batch_size = 1
                                num_inference_steps = 15
                                
                                # Set up scheduler manually
                                pipe.scheduler.set_timesteps(num_inference_steps, device=device)

                                noisy_latents, noise, timesteps, cross_unet, cross_ctrl, added_cond_kwargs, normal_map = prepare_latents_and_conditioning(
                                    batch, pipe, fusion, unet_proj, ctrl_proj, config, device
                                )
                                
                                # Use torch.no_grad to save memory
                                with torch.no_grad():
                                    # Manual denoising loop
                                    for i, t in enumerate(timesteps):
                                        # Scale input according to the scheduler
                                        latent_model_input = pipe.scheduler.scale_model_input(noisy_latents, timestep=t)
                                        
                                        # Get ControlNet conditioning
                                        down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                                            latent_model_input,
                                            t,
                                            encoder_hidden_states=cross_ctrl,
                                            controlnet_cond=normal_map,
                                            added_cond_kwargs=added_cond_kwargs,
                                            return_dict=False,
                                        )

                                        # Predict noise with UNet
                                        noise_pred = pipe.unet(
                                            latent_model_input,
                                            t,
                                            encoder_hidden_states=cross_unet,
                                            down_block_additional_residuals=down_block_res_samples,
                                            mid_block_additional_residual=mid_block_res_sample,
                                            added_cond_kwargs=added_cond_kwargs,
                                        ).sample

                                    # Compute previous noisy sample
                                    noisy_latents = pipe.scheduler.step(noise_pred, t, noisy_latents).prev_sample
                                
                                # Decode latents
                                with torch.no_grad():
                                    latents = 1 / 0.18215 * noisy_latents
                                    image = pipe.vae.decode(latents).sample
                                
                                # Convert to PIL
                                image = (image / 2 + 0.5).clamp(0, 1)
                                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                                image = (image * 255).round().astype("uint8")
                                generated_image = Image.fromarray(image[0])
                                
                                # Rest of visualization and saving code
                                # Fix the dimension mismatch in the tensors
                                original_image = batch["pixel_values"][0:1].to(device)
                                #original_image = (original_image / 2 + 0.5).clamp(0, 1)
                                original_image = original_image[0]  # Remove batch dimension to get [C,H,W]

                                # Convert PIL image to tensor and ensure correct shape [C,H,W]
                                generated_tensor = torch.from_numpy(np.array(generated_image)).permute(2, 0, 1) / 255.0
                                generated_tensor = generated_tensor.to('cpu')  # Move to CPU to save GPU memory

                                # Make sure sample_normal has the right shape [C,H,W] (no batch dimension)
                                sample_normal = sample_normal[0] if sample_normal.dim() == 4 else sample_normal
                                sample_normal = sample_normal.to('cpu')  # Move to CPU to save GPU memory

                                # Now all tensors should be [C,H,W]
                                comparison = torch.cat([
                                    sample_normal,      # Normal map [C,H,W]
                                    original_image.cpu(),  # Original face [C,H,W]
                                    generated_tensor,   # Generated face [C,H,W]
                                ], dim=2)  # Concatenate horizontally

                                # Save the comparison
                                preview_path = output_dir / "previews" / f"preview_e{epoch}_s{global_step}.png"
                                preview_path.parent.mkdir(exist_ok=True, parents=True)
                                save_image(comparison, str(preview_path))
                                accelerator.print(f"✅ Preview saved to {preview_path}")
                        
                                unet_ema.restore(pipe.unet.parameters())
                                controlnet_ema.restore(pipe.controlnet.parameters())
                                fusion_ema.restore(fusion.parameters())
                            except Exception as e:
                                print(f"Preview generation failed: {str(e)}")
                                import traceback
                                print(traceback.format_exc())
                                # Continue with training

                        # Log before zeroing gradients
                        optimizer.zero_grad()

                        global_step += 1

                if global_step >= config["training"]["max_train_steps"]:
                    break
                
            # Validation pass
            try:
                pipe.unet.eval()
                pipe.controlnet.eval()
                pipe.vae.eval()
                total_val_loss = 0
                val_diffusion_losses = []
                val_identity_losses = []

                with torch.no_grad():
                    for batch in val_loader:
                        noisy_latents, noise, timesteps, cross_unet, cross_ctrl, added_cond_kwargs, normal_map = prepare_latents_and_conditioning(
                            batch, pipe, fusion, unet_proj, ctrl_proj, config, accelerator.device
                        )

                        # ControlNet forward pass
                        down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=cross_ctrl,
                            controlnet_cond=normal_map,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )

                        debug_print("\n=== Debug: ControlNet Output ===")
                        debug_print(f"Number of down block samples: {len(down_block_res_samples)}")
                        for i, sample in enumerate(down_block_res_samples):
                            debug_print(f"- Down block {i} shape: {sample.shape}")
                        debug_print(f"Mid block sample shape: {mid_block_res_sample.shape}")

                        try:
                            noise_pred = pipe.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=cross_unet,  # Use 2048-dim for UNet
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs,  # Add this parameter
                            ).sample
                        except Exception as e:
                            print(f"Error in unet: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                            raise

                        try:
                            val_losses = calculate_losses(
                                noise_pred,
                                noise,
                                noise_scheduler,     # your DDPMScheduler
                                timesteps,           # the sampled timesteps
                                noisy_latents,       # the latents you passed into UNet
                                batch["pixel_values"],               # batch["pixel_values"]
                                pipe.vae,            # the VAE from your pipeline
                                global_step,
                                warmup_steps=num_warmup_steps,
                                id_loss_weight=0.5,
                                calc_id_every=5
                            )
                        except Exception as e:
                            print(f"Error in loss calculation: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                            raise
                        
                        val_diffusion_losses.append(val_losses["diffusion_loss"].item())
                        val_identity_losses.append(val_losses["identity_loss"].item())
                        total_val_loss += val_losses["total_loss"].item()

                avg_val_loss = total_val_loss / len(val_loader)
                avg_diffusion_loss = sum(val_diffusion_losses) / len(val_diffusion_losses)
                avg_identity_loss = sum(val_identity_losses) / len(val_identity_losses)

                # Print validation results
                accelerator.print(
                    f"\n=== Validation Results ===\n"
                    f"Epoch: {epoch}/{config['training']['num_epochs']}\n"
                    f"Current Validation Loss: {avg_val_loss:.6f}\n"
                    f"Best Validation Loss: {best_loss:.6f}\n"
                    f"Improvement: {(best_loss - avg_val_loss):.6f} ({'✓' if avg_val_loss < best_loss else '✗'})\n"
                    f"Detailed Losses:\n"
                    f"- Diffusion: {avg_diffusion_loss:.6f}\n"
                    f"- Identity: {avg_identity_loss:.6f}\n"
                    f"Patience Counter: {patience_counter}/{patience}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

                pipe.unet.train()

                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    if accelerator.is_main_process:
                        unet_ema.store(pipe.unet.parameters())
                        controlnet_ema.store(pipe.controlnet.parameters())
                        fusion_ema.store(fusion.parameters())
                        best_model_path = output_dir / "best_model"
                        best_model_path.mkdir(parents=True, exist_ok=True)
                        
                        # Unwrap models
                        unwrapped_unet = accelerator.unwrap_model(pipe.unet)
                        unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
                        
                        # Save LoRA state dicts directly
                        torch.save(unwrapped_unet.state_dict(), best_model_path / "unet_lora.pt")
                        torch.save(unwrapped_controlnet.state_dict(), best_model_path / "controlnet_lora.pt")
                        
                        # Save training state
                        training_state = {
                            "best_loss": best_loss,
                            "config": config,
                            "ema": {
                                "unet": unet_ema.state_dict(),
                                "controlnet": controlnet_ema.state_dict(),
                                "fusion": fusion_ema.state_dict()
                            },
                            "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"
                        }
                        torch.save(training_state, best_model_path / "training_state.pt")

                        CHARACTER_DATA_VOLUME.commit()
                        print("✅ Committed volume changes")
                        
                        unet_ema.restore(pipe.unet.parameters())
                        controlnet_ema.restore(pipe.controlnet.parameters())
                        fusion_ema.restore(fusion.parameters())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

            except Exception as e:
                print(f"Error in validation loop: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                # Continue with training instead of crashing
                pipe.unet.train()

        if accelerator.is_main_process:
            final_path = output_dir / "final_model"
            final_path.mkdir(parents=True, exist_ok=True)
            
            # Unwrap and save final LoRA weights
            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
            
            # Save all model components
            torch.save(unwrapped_unet.state_dict(), final_path / "unet_lora.pt")
            torch.save(unwrapped_controlnet.state_dict(), final_path / "controlnet_lora.pt")
            
            # Save final training state
            training_state = {
                "config": config,
                "ema": {
                    "unet": unet_ema.state_dict(),
                    "controlnet": controlnet_ema.state_dict(),
                    "fusion": fusion_ema.state_dict()
                },
                "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"
            }
            torch.save(training_state, final_path / "training_state.pt")
            CHARACTER_DATA_VOLUME.commit()
            print("✅ Committed volume changes")

        # Add these new debug statements
        debug_print("\nPipeline configuration:")
        debug_print(f"UNet config: {pipe.unet.config}")
        debug_print(f"ControlNet config: {pipe.controlnet.config}")
        debug_print(f"VAE config: {pipe.vae.config}")

        return {
            "status": "success",
            "message": f"Training completed for {character_name}",
            "output_dir": str(output_dir)
        }
    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        return {
            "status": "error",
            "message": error_msg
        }