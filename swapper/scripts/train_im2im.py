#!/usr/bin/env python3
"""
train_mesh2face_lora.py

Fine-tunes a Low-Rank Adapter (LoRA) on top of SDXL’s U-Net
to learn a mesh→photoreal face mapping for your character.
Uses paired mesh renders (input) and FLUX-generated portraits (target).
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLImg2ImgPipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision import transforms as T
from PIL import Image

# ─── 1) CONFIGURATION ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train SDXL mesh→face LoRA")
    p.add_argument("--mesh-dir",    type=Path, default="dataset/meshes",
                   help="Directory of mesh render PNGs")
    p.add_argument("--ref-dir",     type=Path, default="dataset/refs",
                   help="Directory of FLUX reference PNGs")
    p.add_argument("--out-dir",     type=Path, default="sdxl_mesh2face_lora",
                   help="Where to save the trained LoRA weights")
    p.add_argument("--base-model",  type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="SDXL checkpoint for fine-tuning")
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--rank",        type=int, default=4,
                   help="LoRA rank (r)")
    p.add_argument("--alpha",       type=int, default=16,
                   help="LoRA alpha multiplier")
    return p.parse_args()

# ─── 2) DATASET ────────────────────────────────────────────────────────────────
class MeshFaceDataset(Dataset):
    def __init__(self, mesh_dir: Path, ref_dir: Path, transform):
        self.mesh_paths = sorted(mesh_dir.glob("*.png"))
        self.ref_paths  = sorted(ref_dir.glob("*.png"))
        assert len(self.mesh_paths) == len(self.ref_paths), \
            "Mesh and ref counts do not match!"
        self.tf = transform

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_img = Image.open(self.mesh_paths[idx]).convert("RGB")
        ref_img  = Image.open(self.ref_paths[idx]).convert("RGB")
        return self.tf(mesh_img), self.tf(ref_img)

# ─── 3) TRAINING LOOP ──────────────────────────────────────────────────────────
def train_lora(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3.1 Load the SDXL img2img pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    ).to(device)

    # 3.2 Prepare U-Net for LoRA training
    pipe.unet = prepare_model_for_kbit_training(pipe.unet)
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["down_blocks", "mid_block", "up_blocks"],
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    # 3.3 Build DataLoader
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = MeshFaceDataset(args.mesh_dir, args.ref_dir, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 3.4 Optimizer & Scheduler (optional)
    optim = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr)

    # 3.5 Training
    pipe.scheduler = pipe.scheduler  # already configured in the pipeline
    for epoch in range(1, args.epochs + 1):
        pipe.unet.train()
        total_loss = 0.0
        for mesh_imgs, ref_imgs in dl:
            mesh_imgs = mesh_imgs.to(device)
            ref_imgs  = ref_imgs.to(device)
            # Encode to latents
            mesh_latents = pipe.vae.encode(mesh_imgs).latent_dist.sample()
            ref_latents  = pipe.vae.encode(ref_imgs ).latent_dist.sample()

            # Sample timestep & noise
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps,
                                      (mesh_latents.shape[0],),
                                      device=device)
            noise = torch.randn_like(ref_latents)
            noisy_ref = pipe.scheduler.add_noise(ref_latents, noise, timesteps)

            # Forward UNet on noisy mesh latents
            noise_pred = pipe.unet(
                noisy_ref,
                timesteps,
                encoder_hidden_states=pipe.text_encoder(  # embedding of "<rrrdaniel>"
                    torch.tensor([[pipe.tokenizer("<rrrdaniel>", return_tensors="pt")["input_ids"][0]]],
                                 device=device)
                ),
                added_cond_kwargs={"orig_image_latents": mesh_latents},
            ).sample

            # Compute loss & backward
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * mesh_imgs.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch}/{args.epochs} — avg loss: {avg:.4f}")

    # 3.6 Save the LoRA adapters
    os.makedirs(args.out_dir, exist_ok=True)
    pipe.unet.save_pretrained(args.out_dir)
    print(f"LoRA saved to {args.out_dir}")

# ─── 4) ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train_lora(args)
