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
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torchvision import transforms as T
from PIL import Image
import yaml
from accelerate import Accelerator
import datetime
from diffusers.training_utils import EMAModel
from transformers import get_cosine_schedule_with_warmup
import lpips

# ─── 1) CONFIGURATION ─────────────────────────────────────────────────────────

MOUNT_ROOT = "/workspace"
DATA_MOUNT = f"{MOUNT_ROOT}/data/characters"
CACHE_MOUNT = f"{MOUNT_ROOT}/cache"
CHARACTER_DATA_VOLUME = modal.Volume.from_name("character-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)  # Single volume for all caches

# ─── 2) DATASET ────────────────────────────────────────────────────────────────
class MeshFaceDataset(Dataset):
    def __init__(self, character_path: Path, transform):
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

        mesh_latent_dir = character_path / "processed" / "latents" / "mesh"
        ref_latent_dir  = character_path / "processed" / "latents" / "ref"
        self.frame_ids = sorted([p.stem for p in mesh_latent_dir.glob("*.pt") if (ref_latent_dir / f"{p.stem}.pt").exists()])
        self.mesh_latent_dir = mesh_latent_dir
        self.ref_latent_dir = ref_latent_dir

    def __len__(self):
        return len(self.renders)

    def __getitem__(self, idx):
        render_img = Image.open(self.renders[idx]).convert("RGB")
        face_img  = Image.open(self.faces[idx]).convert("RGB")
        frame_id = self.frame_ids[idx]
        mesh_latent = torch.load(self.mesh_latent_dir / f"{frame_id}.pt")
        ref_latent  = torch.load(self.ref_latent_dir  / f"{frame_id}.pt")
        return self.tf(render_img), self.tf(face_img), mesh_latent, ref_latent

# ─── 3) HELPER FUNCTIONS ──────────────────────────────────────────────────────

def get_image_transform(resolution, dtype=torch.float32):
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
        T.Lambda(lambda x: x * 0.5 + 0.5),
        T.ConvertImageDtype(dtype)
    ])

def preencode_latents(pipe, tf, character_path, device="cuda"):
    vae = pipe.vae.eval()

    # Find all pairs
    renders = sorted((character_path / "processed" / "renders").glob("*.png"))
    faces   = sorted((character_path / "processed" / "faces").glob("*.png"))

    mesh_latent_dir = character_path / "processed" / "latents" / "mesh"
    ref_latent_dir  = character_path / "processed" / "latents" / "ref"
    mesh_latent_dir.mkdir(parents=True, exist_ok=True)
    ref_latent_dir.mkdir(parents=True, exist_ok=True)

    for r, f in zip(renders, faces):
        frame_id = r.stem
        mesh_img = tf(Image.open(r).convert("RGB")).to(device)
        ref_img  = tf(Image.open(f).convert("RGB")).to(device)

        with torch.no_grad():
            mesh_latent = (vae.encode(mesh_img.unsqueeze(0)).latent_dist.sample() * vae.config.scaling_factor).squeeze(0)
            ref_latent  = (vae.encode(ref_img.unsqueeze(0)).latent_dist.sample() * vae.config.scaling_factor).squeeze(0)

        torch.save(mesh_latent, mesh_latent_dir / f"{frame_id}.pt")
        torch.save(ref_latent,  ref_latent_dir  / f"{frame_id}.pt")
        print(f"Saved latents for {frame_id}")

def save_training_preview(pipe, batch, output_dir, step, device, config):
    pipe.unet.eval()
    with torch.no_grad():
        pixel_values = batch[0][0:1].to(device)
        target_img   = batch[1][0].cpu()
        output = pipe(
            image=pixel_values,
            prompt="<rrrdaniel>",
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]  # PIL Image

        # Convert tensors to PIL for side-by-side
        def tensor_to_pil(t):
            t = t.clamp(0, 1)
            t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(t)

        input_pil = tensor_to_pil(pixel_values[0])
        target_pil = tensor_to_pil(target_img)
        output_pil = output

        # Concatenate horizontally
        w, h = input_pil.width, input_pil.height
        preview = Image.new("RGB", (w * 3, h))
        preview.paste(input_pil, (0, 0))
        preview.paste(target_pil, (w, 0))
        preview.paste(output_pil, (w * 2, 0))

        preview_dir = Path(output_dir) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"preview_step_{step}.png"
        preview.save(preview_path)
        print(f"✅ Saved training preview to {preview_path}")
    pipe.unet.train()

def training_step(
    pipe,
    batch,
    noise_scheduler,
    device,
    accelerator=None,
    optimizer=None,
    lr_scheduler=None,
    unet_ema=None,
    is_training=True,
    config=None,
    lpips_loss_fn=None,
):
    """
    Performs a single training/validation step.
    
    Args:
        pipe: SDXL pipeline
        batch: Tuple of (mesh_imgs, ref_imgs)
        noise_scheduler: Noise scheduler
        device: Device to run on
        accelerator: Accelerator instance (only needed for training)
        optimizer: Optimizer (only needed for training)
        lr_scheduler: Learning rate scheduler (only needed for training)
        unet_ema: EMA model (only needed for training)
        is_training: Whether this is a training step
        config: Training configuration
        lpips_loss_fn: LPIPS loss function (only needed for training)
    
    Returns:
        dict: Loss values and predictions
    """
    mesh_imgs, ref_imgs, mesh_latents, ref_latents = batch
    mesh_imgs = mesh_imgs.to(device)
    ref_imgs = ref_imgs.to(device)
    mesh_latents = mesh_latents.to(device)
    ref_latents = ref_latents.to(device)

    # 3) sample noise & timesteps
    noise = torch.randn_like(ref_latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (mesh_latents.shape[0],),
        device=device,
    )
    noisy_ref = noise_scheduler.add_noise(ref_latents, noise, timesteps)

    # 4) prompt → cross-attention & pooled embeddings
    seq_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
        ["<rrrdaniel>"] * mesh_latents.shape[0],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # 5) micro-conditioning (optional; drop if you want simpler)
    H, W = mesh_imgs.shape[-2:]
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
    add_time_ids     = add_time_ids.repeat(mesh_latents.shape[0], 1).to(device)
    add_neg_time_ids = add_neg_time_ids.repeat(mesh_latents.shape[0], 1).to(device)

    # 6) cast everything to half if your UNet is fp16
    """mesh_latents     = mesh_latents.float()
    noisy_ref        = noisy_ref.float()
    seq_embeds       = seq_embeds.float()
    pooled_embeds    = pooled_embeds.float()
    add_time_ids     = add_time_ids.float()
    add_neg_time_ids = add_neg_time_ids.float()"""

    #print("dtype debug:")
    #print(mesh_latents.dtype)
    #print(noisy_ref.dtype)
    #print(seq_embeds.dtype)
    #print(pooled_embeds.dtype)
    #print(add_time_ids.dtype)
    #print(add_neg_time_ids.dtype)

    with torch.autograd.set_detect_anomaly(True):
        with accelerator.autocast():
            # 7) forward
            noise_pred = pipe.unet(
                noisy_ref,
                timesteps,
                encoder_hidden_states=seq_embeds,
                added_cond_kwargs={
                    "orig_image_latents": mesh_latents,      # mesh→output latents
                    "text_embeds":        pooled_embeds,     # pooled prompt embedding
                    "time_ids":           add_time_ids,      # micro-conditioning
                    "neg_time_ids":       add_neg_time_ids,  # negative-aesthetic branch
                },
            ).sample

            # 8) loss
            mse_loss = F.mse_loss(noise_pred, noise)

            # --- LPIPS loss ---
            # Decode latents to images for LPIPS
            with torch.no_grad():
                # [B, C, H, W] in [-1, 1] for LPIPS
                decoded_pred = pipe.vae.decode(noise_pred).sample
                decoded_target = pipe.vae.decode(noise).sample

            # Clamp and scale to [-1, 1] for LPIPS
            decoded_pred = (decoded_pred / 2).clamp(-1, 1)
            decoded_target = (decoded_target / 2).clamp(-1, 1)

            lpips_loss = lpips_loss_fn(decoded_pred, decoded_target).mean()

            # Weighted total loss
            loss = 0.7 * mse_loss + 0.3 * lpips_loss

    if is_training:
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        #accelerator.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
        total_norm = torch.norm(
            torch.stack([g.detach().norm() for g in pipe.unet.parameters() if g.grad is not None])
        )
        #print("Gradient norm:", total_norm.item())  
        optimizer.step()
        lr_scheduler.step()
        #for n, p in pipe.unet.named_parameters():
        #    if "lora_" in n:
        #        print(f"{n}: mean={p.data.mean():.3e}, std={p.data.std():.3e}")

        #scaler.update()
        optimizer.zero_grad()
        if accelerator.sync_gradients and unet_ema is not None:
            unet_ema.step(pipe.unet.parameters())
    else:
        total_norm = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": loss,
        "mse_loss": mse_loss,
        "lpips_loss": lpips_loss,
        "noise_pred": noise_pred,
        "noise": noise,
        "mesh_latents": mesh_latents,
        "ref_latents": ref_latents,
        "total_norm": total_norm
    }

# ─── 4) TRAINING LOOP ──────────────────────────────────────────────────────────
def train_lora(character_name: str, output_dir: Optional[str] = None, from_checkpoint: bool = False):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
    sdxl_cache_path = os.path.join(modal_cache, "huggingface", "sdxl-base-1.0")
        
    # Load config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    accelerator = Accelerator(
        mixed_precision=config["optimization"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
    )

    print(f"Using device: {device}")
    # Load the SDXL img2img pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        sdxl_cache_path,
        torch_dtype=torch.float32,
    ).to(device)
    #pipe.unet = torch.compile(pipe.unet)
    #pipe.unet = pipe.unet.half()
    
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.unet.enable_gradient_checkpointing()
    #pipe.enable_xformers_memory_efficient_attention()   # if you have xformers installed

    # Prepare U-Net for LoRA training
    lora_cfg = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=[
            "conv1", "conv2",       # ResNet blocks
            "conv_shortcut",        # any skip-connections
            "to_q", "to_k", "to_v", # attention
            "proj_in", "proj_out"
        ],
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    pipe.unet = pipe.unet.to(device)


    num_added = pipe.tokenizer.add_special_tokens({"additional_special_tokens":["<rrrdaniel>"]})
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer))

    # Build DataLoader
    transform = get_image_transform(config["resolution"])
    
    # Initialize dataset and split into train/val
    preencode_latents(pipe, transform, character_path, device)
    dataset = MeshFaceDataset(character_path, transform)
    val_size = int(len(dataset) * 0.2)  # 20% for validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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

    # Set the transform for both train and val loaders
    train_loader.dataset.dataset.tf = transform
    val_loader.dataset.dataset.tf = transform

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        sdxl_cache_path,
        subfolder="scheduler"
    )

    # Initialize EMA
    unet_ema = EMAModel(
        pipe.unet.parameters(),
        decay=0.9999,
        model_cls=pipe.unet.__class__,
        device=device
    )

    # Prepare models with accelerator
    pipe.unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    # Initialize LPIPS loss function
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_loss_fn.eval()

    # Training loop
    best_loss = float('inf')
    patience = config["training"].get("patience", 5)
    patience_counter = 0

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        pipe.unet.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate():
                try:
                    step_output = training_step(
                        pipe=pipe,
                        batch=batch,
                        noise_scheduler=noise_scheduler,
                        device=device,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        unet_ema=unet_ema,
                        is_training=True,
                        config=config,
                        lpips_loss_fn=lpips_loss_fn
                    )
                except Exception as e:
                    print(f"Error in training step: {e}")
                    raise e
                
                total_loss += step_output["loss"].item()
                #print("total_loss debug:")
                #print(total_loss)

                # Logging
                if global_step % config["training"].get("log_steps", 10) == 0:
                    accelerator.print(
                        f"\n=== Training Statistics ===\n"
                        f"Epoch: {epoch}/{config['training']['num_epochs']} | "
                        f"Step: {global_step}/{num_training_steps} | "
                        f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
                        f"Gradient Norm: {step_output['total_norm']:.4f}\n"
                        f"Losses:\n"
                        f"- Total: {step_output['loss']:.4f}\n"
                        f"- MSE:   {step_output['mse_loss']:.4f}\n"
                        f"- LPIPS: {step_output['lpips_loss']:.4f}\n"
                    )

                # Save training preview
                if global_step % config["training"].get("preview_steps", 50) == 0:
                    save_training_preview(pipe, batch, output_dir, global_step, accelerator.device, config)

                # Save checkpoint
                if global_step % config["training"].get("save_steps", 500) == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = output_dir / "checkpoints" / f"checkpoint-{epoch}-{global_step}"
                        checkpoint_path.mkdir(parents=True, exist_ok=True)
                        
                        # Save model state
                        pipe.unet.save_pretrained(checkpoint_path / "unet_lora.pt")
                        
                        print("Saving model state")
                        # Save training state
                        training_state = {
                            "step": global_step,
                            "epoch": epoch,
                            "config": config,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                            "ema": unet_ema.state_dict()
                        }
                        torch.save(training_state, checkpoint_path / "training_state.pt")
                        
                        CHARACTER_DATA_VOLUME.commit()
                        print("✅ Committed volume changes")

                global_step += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Validation
        pipe.unet.eval()
        total_val_loss = 0.0
        
        # Store training weights and load EMA weights for validation
        if unet_ema is not None:
            unet_ema.store(pipe.unet.parameters())
            unet_ema.copy_to(pipe.unet.parameters())
        
        with torch.no_grad():
            for batch in val_loader:
                step_output = training_step(
                    pipe=pipe,
                    batch=batch,
                    noise_scheduler=noise_scheduler,
                    device=device,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    unet_ema=unet_ema,
                    is_training=False,
                    config=config,
                    lpips_loss_fn=lpips_loss_fn
                )
                total_val_loss += step_output["loss"].item()

        # Restore training weights if using EMA
        if unet_ema is not None:
            unet_ema.restore(pipe.unet.parameters())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            if accelerator.is_main_process:
                unet_ema.store(pipe.unet.parameters())
                best_model_path = output_dir / "best_model"
                best_model_path.mkdir(parents=True, exist_ok=True)
                
                # Save LoRA state dict
                torch.save(pipe.unet.state_dict(), best_model_path / "unet_lora.pt")
                
                # Save training state
                training_state = {
                    "best_loss": best_loss,
                    "config": config,
                    "ema": unet_ema.state_dict(),
                    "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"
                }
                torch.save(training_state, best_model_path / "training_state.pt")
                
                CHARACTER_DATA_VOLUME.commit()
                print("✅ Committed volume changes")
                
                unet_ema.restore(pipe.unet.parameters())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    if accelerator.is_main_process:
        final_path = output_dir / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Save final LoRA weights
        torch.save(pipe.unet.state_dict(), final_path / "unet_lora.pt")
        
        # Save final training state
        training_state = {
            "config": config,
            "ema": unet_ema.state_dict(),
            "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"
        }
        torch.save(training_state, final_path / "training_state.pt")
        CHARACTER_DATA_VOLUME.commit()
        print("✅ Committed volume changes")

    return {
        "status": "success",
        "message": f"Training completed for {character_name}",
        "output_dir": str(output_dir)
    }