# scripts/train_lora.py

import os
import json
import torch
import yaml
import modal
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, CosineEmbeddingLoss, functional as F
from transformers import get_cosine_schedule_with_warmup
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    ControlNetModel,
    DDPMScheduler,
)
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator

from swapper.utils.image_utils import preprocess_image
from swapper.utils.embedding_utils import get_face_embedding

from models.lighting_mlp import LightingMLP

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

# --------------- Dataset ----------------
# --- Dataset ---
class CharacterDataset(Dataset):
    def __init__(self, character_dir, config):
        print(f"\n=== Initializing CharacterDataset ===")
        print(f"Character directory: {character_dir}")
        
        self.config = config
        self.faces_dir = os.path.join(character_dir, "processed/maps/faces")
        self.normals_dir = os.path.join(character_dir, "processed/maps/normals")
        self.meta_path = os.path.join(character_dir, "metadata.json")
        
        print(f"Checking paths:")
        print(f"- Faces dir exists: {os.path.exists(self.faces_dir)}")
        print(f"- Normals dir exists: {os.path.exists(self.normals_dir)}")
        print(f"- Metadata exists: {os.path.exists(self.meta_path)}")

        with open(self.meta_path, "r") as f:
            self.metadata = json.load(f)["frames"]
        
        print(f"\nProcessing samples...")
        self.samples = []
        for fname in os.listdir(self.faces_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                base = os.path.splitext(fname)[0]
                if base not in self.metadata:
                    print(f"Warning: {base} found in faces dir but not in metadata")
                    continue
                sample = {
                    "image": os.path.join(self.faces_dir, fname),
                    "normal_map": os.path.join(self.normals_dir, fname),
                    "lighting": self.metadata[base].get("lighting"),
                    "embedding": self.metadata[base].get("embedding"),
                    "frame_id": base,
                }
                self.samples.append(sample)
        
        print(f"Found {len(self.samples)} valid samples")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = preprocess_image(sample["image"], target_size=(self.config["resolution"], self.config["resolution"]))
        normal = preprocess_image(sample["normal_map"], target_size=(self.config["controlnet_resolution"], self.config["controlnet_resolution"]))
        normal_tensor = torch.from_numpy(normal).permute(2, 0, 1)  # (3, H, W)

        embedding = sample["embedding"]
        if embedding is None:
            embedding = get_face_embedding(sample["image"])
        embedding = torch.tensor(embedding, dtype=torch.float32)

        lighting = torch.tensor(sample["lighting"], dtype=torch.float32)

        return {
            "pixel_values": torch.tensor(image).permute(2, 0, 1),
            "normal_map": normal_tensor,
            "embedding": embedding,
            "lighting": lighting,
        }

    def __len__(self):
        return len(self.samples)

# --------------- Helper functions ----------------

def prepare_latents(vae, image_tensor, scheduler):
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, noise, timesteps

# --------------- Training ----------------
def train(character_name: str):
    try:
        print("\n=== Starting training setup ===")
        print(f"Current working directory: {os.getcwd()}")
        
        # Use the Modal cache path for the model
        modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
        landmarks_path = os.path.join(modal_cache, "models", "shape_predictor_68_face_landmarks.dat")
        
        print(f"\nChecking face landmarks model:")
        print(f"- Modal cache directory: {modal_cache}")
        print(f"- Looking for: {landmarks_path}")
        print(f"- Path exists: {os.path.exists(landmarks_path)}")
        if os.path.exists(landmarks_path):
            print(f"- Is symlink: {os.path.islink(landmarks_path)}")
            print(f"- Real path: {os.path.realpath(landmarks_path)}")
        else:
            raise RuntimeError(f"Face landmarks model not found at {landmarks_path}")
        
        # 2. Set environment variable for dlib
        os.environ["DLIB_SHAPE_PREDICTOR"] = os.path.abspath(landmarks_path)
        print(f"\nSet DLIB_SHAPE_PREDICTOR to: {os.environ['DLIB_SHAPE_PREDICTOR']}")
        
        # 3. Now safe to initialize dataset
        with open("configs/train_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        print("\nLoading dataset...")
        character_path = Path(config["base_dir"]) / character_name
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
            
        accelerator = Accelerator(
            mixed_precision=config["optimization"]["mixed_precision"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
        )

        output_dir = Path("/data/characters") / character_name / "lora_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

        # Get the cache path from environment
        modal_cache = os.getenv("MODAL_CACHE_DIR", "/workspace/cache")
        sdxl_cache_path = os.path.join(modal_cache, "huggingface", "sdxl-base-1.0")

        print(f"\nLoading SDXL model from cache:")
        print(f"- Cache path: {sdxl_cache_path}")
        print(f"- Path exists: {os.path.exists(sdxl_cache_path)}")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            sdxl_cache_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16"
        )
        # Enable gradient checkpointing on specific components
        pipe.unet.enable_gradient_checkpointing()
        pipe.vae.enable_gradient_checkpointing()

        # LoRA to UNet
        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"].get("lora_alpha", 32),
            lora_dropout=config["model"].get("lora_dropout", 0.1),
            target_modules=config["model"]["target_modules"],
            bias="none",
            task_type="CONDITIONING"  # Changed from CAUSAL_LM
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        pipe.unet.enable_gradient_checkpointing()

        # ControlNet (LoRA or full)
        pipe.controlnet = ControlNetModel.from_unet(
            pipe.unet,
            # Remove any LoRA config here if you're passing it
        )

        # Lighting MLP
        lighting_mlp = LightingMLP(input_dim=9, output_dim=128).to(accelerator.device)

        # Add to training setup
        num_training_steps = len(dataloader) * config["training"]["num_epochs"]
        num_warmup_steps = num_training_steps // 10

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        optimizer = torch.optim.Adam(
            list(pipe.unet.parameters()) +
            list(pipe.controlnet.parameters()) +
            list(lighting_mlp.parameters()),
            lr=config["training"]["learning_rate"]
        )

        loss_fn = MSELoss()
        cosine_loss_fn = CosineEmbeddingLoss()

        # Prepare models and data for distributed training
        pipe.unet, lighting_mlp, optimizer, dataloader = accelerator.prepare(
            pipe.unet, lighting_mlp, optimizer, dataloader
        )

        # Add EMA
        from diffusers.training_utils import EMAModel
        ema = EMAModel(
            pipe.unet.parameters(),
            decay=0.9999,
            model_cls=pipe.unet.__class__
        )
        
        # Add validation
        best_loss = float('inf')
        patience = config["training"].get("patience", 5)
        patience_counter = 0

        for epoch in range(config["training"]["num_epochs"]):
            total_loss = 0
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(pipe.unet):
                    with torch.cuda.amp.autocast():
                        # Inputs
                        image = batch["pixel_values"].to(accelerator.device)
                        normal_map = batch["normal_map"].to(accelerator.device)
                        identity = F.normalize(batch["embedding"].to(accelerator.device), dim=-1)
                        lighting = F.normalize(batch["lighting"].to(accelerator.device), dim=-1)
                        lighting_emb = lighting_mlp(lighting)

                        combined_emb = identity + lighting_emb

                        # Noise prediction
                        noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, scheduler)
                        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=combined_emb,
                                            controlnet_cond=normal_map.unsqueeze(1)).sample

                        diffusion_loss = loss_fn(noise_pred, noise)
                        cosine_loss = cosine_loss_fn(combined_emb, identity, torch.ones(identity.size(0)).to(accelerator.device))
                        total_loss = diffusion_loss + config["training"].get("lambda_identity", 0.5) * cosine_loss

                        accelerator.backward(total_loss)
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()

                    # Update EMA
                    ema.step(pipe.unet.parameters())

                    if step % config["training"].get("log_steps", 10) == 0:
                        accelerator.print(
                            f"Epoch {epoch} | Step {step} | "
                            f"Loss: {total_loss.item():.4f} | "
                            f"Diffusion: {diffusion_loss.item():.4f} | "
                            f"Identity: {cosine_loss.item():.4f}"
                        )
                    
                    # Save checkpoints
                    if step % config["training"].get("save_steps", 500) == 0:
                        if accelerator.is_main_process:
                            save_path = output_dir / f"checkpoint-{epoch}-{step}"
                            state_dict = {
                                "step": step,
                                "epoch": epoch,
                                "model_state": accelerator.get_state_dict(pipe.unet),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "ema": ema.state_dict(),
                                "config": config,
                            }
                            accelerator.save(state_dict, save_path / "checkpoint.pt")

                if step >= config["training"]["max_train_steps"]:
                    break
                
            # Validation and early stopping
            avg_loss = total_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                if accelerator.is_main_process:
                    ema.store(pipe.unet.parameters())
                    pipe.save_pretrained(output_dir / "best_model")
                    ema.restore(pipe.unet.parameters())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        pipe.save_pretrained(str(output_dir))

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