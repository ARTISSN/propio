# scripts/train_lora.py

import os
import re
import json
import torch
import yaml
import modal
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import MSELoss, CosineEmbeddingLoss, functional as F
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

from swapper.utils.image_utils import preprocess_image
from swapper.utils.embedding_utils import get_face_embedding

from models.lighting_mlp import LightingMLP

# Debug configuration
DEBUG_MODE = True  # Set to True to enable detailed debugging output

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
                    "embedding": frame_data.get("embedding"),
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
def load_checkpoint(checkpoint_path: Path, pipe, lighting_to_cross, lighting_to_pooled, identity_to_cross, identity_to_pooled, optimizer, lr_scheduler, ema):
    """Load model and training state from checkpoint."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    try:
        # Load model weights
        unet_path = checkpoint_path / "unet_lora.pt"
        controlnet_path = checkpoint_path / "controlnet_lora.pt"
        lighting_cross_path = checkpoint_path / "lighting_to_cross.pt"
        lighting_pooled_path = checkpoint_path / "lighting_to_pooled.pt"
        identity_cross_path = checkpoint_path / "identity_to_cross.pt"
        identity_pooled_path = checkpoint_path / "identity_to_pooled.pt"
        training_state_path = checkpoint_path / "training_state.pt"
        
        required_files = [
            unet_path, controlnet_path, lighting_cross_path, lighting_pooled_path,
            identity_cross_path, identity_pooled_path, training_state_path
        ]
        
        if not all(p.exists() for p in required_files):
            raise FileNotFoundError("Checkpoint files are incomplete")
            
        # Load model weights
        pipe.unet.load_state_dict(torch.load(unet_path))
        pipe.controlnet.load_state_dict(torch.load(controlnet_path))
        lighting_to_cross.load_state_dict(torch.load(lighting_cross_path))
        lighting_to_pooled.load_state_dict(torch.load(lighting_pooled_path))
        identity_to_cross.load_state_dict(torch.load(identity_cross_path))
        identity_to_pooled.load_state_dict(torch.load(identity_pooled_path))
        
        # Load training state
        training_state = torch.load(training_state_path)
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(training_state["optimizer"])
        lr_scheduler.load_state_dict(training_state["scheduler"])
        
        # Load EMA state
        if "ema" in training_state:
            ema.load_state_dict(training_state["ema"])
        
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
    
    if validation_inputs["pooled_prompt_embeds"] is not None:
        if validation_inputs["pooled_prompt_embeds"].ndim != 2:
            raise ValueError(f"pooled_prompt_embeds must be 2-dimensional, got {validation_inputs['pooled_prompt_embeds'].ndim} dimensions")
        if validation_inputs["pooled_prompt_embeds"].shape[-1] != 1280:
            raise ValueError(f"pooled_prompt_embeds must have shape [..., 1280], got [..., {validation_inputs['pooled_prompt_embeds'].shape[-1]}]")

    # 5. Check device consistency
    if validation_inputs["prompt_embeds"] is not None and validation_inputs["pooled_prompt_embeds"] is not None:
        if validation_inputs["prompt_embeds"].device != validation_inputs["pooled_prompt_embeds"].device:
            raise ValueError("prompt_embeds and pooled_prompt_embeds must be on the same device")

    # 6. Check dtype consistency
    if validation_inputs["prompt_embeds"] is not None:
        if validation_inputs["prompt_embeds"].dtype != torch.bfloat16:
            raise ValueError(f"prompt_embeds must be of dtype bfloat16, got {validation_inputs['prompt_embeds'].dtype}")

    return True


def train(character_name: str, output_dir: Optional[str] = None, from_checkpoint: bool = False):
    """Train the model with support for checkpoints and multiple runs."""
    try:
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
        
        print(f"\nChecking face landmarks model:")
        print(f"- Modal cache directory: {modal_cache}")
        print(f"- Looking for: {landmarks_path}")
        print(f"- Path exists: {os.path.exists(landmarks_path)}")
        if os.path.exists(landmarks_path):
            print(f"- Is symlink: {os.path.islink(landmarks_path)}")
            print(f"- Real path: {os.path.realpath(landmarks_path)}")
            # Set environment variable for dlib with the absolute path
            os.environ["DLIB_SHAPE_PREDICTOR"] = os.path.abspath(landmarks_path)
            print(f"\nSet DLIB_SHAPE_PREDICTOR to: {os.environ['DLIB_SHAPE_PREDICTOR']}")
        else:
            raise RuntimeError(f"Face landmarks model not found at {landmarks_path}")
        
        # 2. Set environment variable for dlib
        os.environ["DLIB_SHAPE_PREDICTOR"] = os.path.abspath(landmarks_path)
        print(f"\nSet DLIB_SHAPE_PREDICTOR to: {os.environ['DLIB_SHAPE_PREDICTOR']}")
        
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
            # 1) load the base SDXL pipeline
            base = StableDiffusionXLPipeline.from_pretrained(
            sdxl_cache_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
                variant="bf16",
            ).to(accelerator.device)

            # 2) clone its UNet into a new ControlNet
            controlnet = ControlNetModel.from_unet(base.unet)

            # Modify the config after creation
            controlnet.config.cross_attention_dim = 2048  # Match UNet's dimension
            controlnet.config.conditioning_embedding_out_channels = 2048  # Match cross attention dim

            # Verify the config after modification
            print("\nControlNet config after modification:")
            print(f"cross_attention_dim: {controlnet.config.cross_attention_dim}")
            print(f"conditioning_embedding_out_channels: {controlnet.config.conditioning_embedding_out_channels}")

            # Then create the pipeline with the modified controlnet
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
        
            #for name, module in pipe.controlnet.named_modules():
            #    print(name)
        except Exception as e:
            print(f"Error loading SDXL model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # Add this before creating the LoRA config
        print("\nAvailable modules in ControlNet:")
        for name, _ in pipe.controlnet.named_modules():
            print(f"- {name}")

        # First, create the LoRA config with the correct module names
        controlnet_lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                # Conditioning embedding
                "controlnet_cond_embedding.conv_in",
                "controlnet_cond_embedding.conv_out",
                # Down blocks
                "down_blocks.0.resnets.0.conv1",
                "down_blocks.0.resnets.0.conv2",
                "down_blocks.0.resnets.1.conv1",
                "down_blocks.0.resnets.1.conv2",
                "down_blocks.1.resnets.0.conv1",
                "down_blocks.1.resnets.0.conv2",
                "down_blocks.1.resnets.1.conv1",
                "down_blocks.1.resnets.1.conv2",
                "down_blocks.2.resnets.0.conv1",
                "down_blocks.2.resnets.0.conv2",
                "down_blocks.2.resnets.1.conv1",
                "down_blocks.2.resnets.1.conv2",
                # Mid block
                "mid_block.resnets.0.conv1",
                "mid_block.resnets.0.conv2",
                "mid_block.resnets.1.conv1",
                "mid_block.resnets.1.conv2",
                # Attention layers
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k",
                "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v",
            ],
            bias="none",
            task_type="CONTROLNET"
        )

        # Then verify the target modules
        def verify_target_modules(model, target_modules):
            available_modules = dict(model.named_modules())
            found_modules = []
            missing_modules = []
            
            for target in target_modules:
                if target in available_modules:
                    found_modules.append(target)
                else:
                    missing_modules.append(target)
            
            print(f"\nModule verification results:")
            print(f"Found {len(found_modules)} of {len(target_modules)} target modules")
            
            return len(missing_modules) == 0

        # Verify the configuration
        if not verify_target_modules(pipe.controlnet, controlnet_lora_config.target_modules):
            print("\nWarning: Some target modules were not found. Please check the module names.")
        else:
            print("\nAll target modules verified successfully!")

        # Now create the PEFT model
        try:
            pipe.controlnet = get_peft_model(pipe.controlnet, controlnet_lora_config)
            pipe.controlnet.train()
        except Exception as e:
            print(f"Error creating PEFT model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        try:
            pipe.controlnet.to(accelerator.device)
            pipe.controlnet.enable_gradient_checkpointing()
        except Exception as e:
            print(f"Error getting PEFT model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # After ControlNet LoRA initialization
        debug_print("\n=== Debug: ControlNet LoRA Setup ===")
        debug_print("ControlNet LoRA config:")
        debug_print(f"- Target modules: {controlnet_lora_config.target_modules}")
        debug_print(f"- LoRA rank: {controlnet_lora_config.r}")
        debug_print(f"- LoRA alpha: {controlnet_lora_config.lora_alpha}")

        # Check which modules actually got LoRA weights
        debug_print("\nControlNet modules with LoRA:")
        for name, module in pipe.controlnet.named_modules():
            if hasattr(module, 'lora_layer'):
                debug_print(f"- {name}")

        # 6) check one of your LoRA grads
        try:
            for name, p in pipe.controlnet.named_parameters():
                if "lora_" in name:
                    print(f"{name:40s} → requires_grad={p.requires_grad}")

            cnet = pipe.controlnet
            c_emb = pipe.controlnet.controlnet_cond_embedding
            cnet.train()

            # Freeze non‑LoRA
            for n,p in c_emb.named_parameters():
                p.requires_grad = ("lora_" in n)

            # Check that we actually replaced convs
            print("conv_in type:", type(c_emb.conv_in))
            print("conv_out type:", type(c_emb.conv_out))

            # Dummy forward/backward
            dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=c_emb.conv_in.weight.device)
            out   = pipe.controlnet.controlnet_cond_embedding(dummy)      # [1,C',H',W']
            loss  = out.abs().mean()
            loss.backward()

            # Inspect grads
            found = False
            for n, p in pipe.controlnet.controlnet_cond_embedding.named_parameters():
                if "lora_A" in n:
                    print(f"{n:60s} → grad norm = {p.grad.norm():.3e}")
                    found = True
                    break

            if not found:
                print("❌ zero grads on every LoRA weight — something still isn't hooked up")
        except Exception as e:
            print(f"Error getting PEFT model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        # DEBUG: how many params are trainable?
        total = sum(p.numel() for p in pipe.controlnet.parameters())
        trainable = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
        print(f"🔍 ControlNet params: {total:,}, trainable: {trainable:,}")

        # And print their names:
        for name, p in pipe.controlnet.named_parameters():
            if p.requires_grad:
                print("  🟢", name)

        # Add these debug statements after loading the SDXL pipeline
        print("\n=== Debugging Pipeline Components ===")
        print(f"Pipeline components loaded: {pipe is not None}")
        print(f"Text Encoder 2 exists: {hasattr(pipe, 'text_encoder_2')}")
        if hasattr(pipe, 'text_encoder_2'):
            print(f"Text Encoder 2 config exists: {hasattr(pipe.text_encoder_2, 'config')}")
            if hasattr(pipe.text_encoder_2, 'config'):
                print(f"Text Encoder 2 config: {pipe.text_encoder_2.config}")
                print(f"Projection dim: {getattr(pipe.text_encoder_2.config, 'projection_dim', None)}")

        # Debug config
        print("\n=== Debugging Config ===")
        print(f"Config type: {type(config)}")
        print(f"Config contents: {json.dumps(config, indent=2)}")
        print(f"Model section exists: {'model' in config}")
        if 'model' in config:
            print(f"Model config: {json.dumps(config['model'], indent=2)}")
            print(f"hidden_dim exists: {'hidden_dim' in config['model']}")
            print(f"lighting_dim exists: {'lighting_dim' in config['model']}")

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
            
            # Continue with model initialization...
            hidden_dim = config["model"]["hidden_dim"]
            
        except Exception as e:
            print(f"\n❌ Error during initialization:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

        pipe.vae = pipe.vae.to(dtype=torch.bfloat16) 
        pipe.vae = accelerator.prepare(pipe.vae)
        # Enable gradient checkpointing on specific components
        #pipe.unet.enable_gradient_checkpointing()
        pipe.vae.disable_gradient_checkpointing()

        num = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
        print("ControlNet trainable params:", num)

        # list every submodule whose name contains "attn2"
        attn2_modules = [n for n,_ in pipe.unet.named_modules() if "attn2" in n]
        print(f"Found {len(attn2_modules)} cross‑attention modules:\n", attn2_modules[:10], "…")

        all_mods = [name for name,_ in pipe.unet.named_modules()]
        for pat in [r"attn2\.to_q$", r"attn2\.to_k$", r"attn2\.to_v$", r"attn2\.to_out\.0$"]:
            hits = [n for n in all_mods if re.search(pat, n)]
            print(f"{pat} → {len(hits)} matches")

        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ]
        for pat in target_modules:
            hits = [n for n,_ in pipe.unet.named_modules() if re.search(pat, n)]
            debug_print(f"{pat} → {len(hits)} modules will be LoRA‑wrapped")

        # LoRA config
        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"].get("lora_alpha", 32),
            lora_dropout=config["model"].get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias="none",
            task_type="UNET"
        )

        debug_print([name for name,_ in pipe.unet.named_modules() if re.search(r"down_blocks\.\d+\.attentions", name)])

        # Now LoRA to UNet
        try:
            pipe.unet = get_peft_model(pipe.unet, lora_config)
        except Exception as e:
            print(f"Error getting PEFT model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        pipe.unet.get_base_model().disable_gradient_checkpointing()

        print("❗ UNet trainable params:", 
            sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad))
        print("❗ ControlNet trainable params:", 
            sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad))

        # Lighting MLP
        cross_attention_dim = pipe.unet.config.cross_attention_dim  # should be 2048
        pooled_dim          = pipe.text_encoder_2.config.projection_dim  # 1280
        ctrl_dim = pipe.controlnet.config.projection_class_embeddings_input_dim  # Should be 2816, not 1024
        
        # maps your 128‑dim face embedding → cross‑attention space
        identity_to_cross = torch.nn.Sequential(
            torch.nn.Linear(128, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cross_attention_dim),
        )
        # maps your 128‑dim lighting → cross‑attention space
        lighting_to_cross = torch.nn.Sequential(
            torch.nn.Linear(config["model"]["lighting_dim"], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cross_attention_dim),
        )

        # you need two more heads:
        identity_to_pooled = torch.nn.Sequential(
            torch.nn.Linear(128, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, pooled_dim),
        )
        lighting_to_pooled = torch.nn.Sequential(
            torch.nn.Linear(config["model"]["lighting_dim"], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, pooled_dim),
        )

        optimizer = torch.optim.Adam(
            list(pipe.unet.parameters()) +
            [p for p in pipe.controlnet.parameters() if p.requires_grad] +
            list(lighting_to_cross.parameters()) +
            list(lighting_to_pooled.parameters()) +
            list(identity_to_cross.parameters()) +
            list(identity_to_pooled.parameters()),
            lr=float(config["training"]["learning_rate"])
        )

        # Add to training setup
        num_training_steps = len(train_loader) * config["training"]["num_epochs"]
        num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])

        # Learning rate scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Noise scheduler for diffusion
        noise_scheduler = DDPMScheduler.from_pretrained(
            sdxl_cache_path,
            subfolder="scheduler"
        )

        loss_fn = MSELoss()
        cosine_loss_fn = CosineEmbeddingLoss()

        # Initialize EMA
        ema = EMAModel(
            pipe.unet.parameters(),
            decay=0.9999,
            model_cls=pipe.unet.__class__
        )
        
        # Load checkpoint state if available
        if from_checkpoint and latest_checkpoint:
            checkpoint_state = load_checkpoint(
                latest_checkpoint,
                pipe,
                lighting_to_cross,
                lighting_to_pooled,
                identity_to_cross,
                identity_to_pooled,
                optimizer,
                lr_scheduler,
                ema
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

        # Prepare models AFTER loading checkpoint
        pipe.unet, pipe.controlnet, lighting_to_cross, lighting_to_pooled, \
        identity_to_cross, identity_to_pooled, \
        optimizer, train_loader, val_loader = accelerator.prepare(
            pipe.unet, pipe.controlnet, lighting_to_cross, lighting_to_pooled,
            identity_to_cross, identity_to_pooled,
            optimizer, train_loader, val_loader
        )

        # Add this after ControlNet initialization
        print("\n=== Debug: Model Dimensions ===")
        print(f"UNet cross attention dim: {pipe.unet.config.cross_attention_dim}")
        print(f"ControlNet cross attention dim: {pipe.controlnet.config.cross_attention_dim}")
        print(f"ControlNet projection dim: {pipe.controlnet.config.projection_class_embeddings_input_dim}")

        # Also check the actual attention blocks
        for name, module in pipe.controlnet.named_modules():
            if "attn" in name:
                print(f"\nAttention module: {name}")
                if hasattr(module, "to_q"):
                    print(f"Query dim: {module.to_q.in_features}")
                if hasattr(module, "to_k"):
                    print(f"Key dim: {module.to_k.in_features}")
                if hasattr(module, "to_v"):
                    print(f"Value dim: {module.to_v.in_features}")

        # Training loop
        best_loss = float('inf')
        patience = config["training"].get("patience", 5)
        patience_counter = 0

        def init_weights(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        pipe.controlnet.apply(init_weights)

        for epoch in range(start_epoch, config["training"]["num_epochs"]):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                with accelerator.accumulate():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Inputs
                        image = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
                        normal_map = batch["normal_map"].to(accelerator.device, dtype=torch.bfloat16)
                        identity = F.normalize(batch["embedding"].to(accelerator.device), dim=-1)
                        lighting = F.normalize(batch["lighting"].to(accelerator.device), dim=-1)

                        # For UNet - use 2048-dim cross attention
                        id_cross = identity_to_cross(identity)      # [B, 2048]
                        light_cross = lighting_to_cross(lighting)   # [B, 2048]
                        combined_cross = (id_cross + light_cross).unsqueeze(1)  # [B,1,2048]

                        # Before ControlNet forward pass
                        debug_print("\n=== Debug: Tensor Dimensions ===")
                        debug_print(f"combined_cross shape: {combined_cross.shape}")
                        debug_print(f"ControlNet cross attention dim: {pipe.controlnet.config.cross_attention_dim}")
                        debug_print(f"UNet cross attention dim: {pipe.unet.config.cross_attention_dim}")

                        # Add dimension check
                        assert combined_cross.shape[-1] == pipe.controlnet.config.cross_attention_dim, \
                            f"Expected cross attention dim {pipe.controlnet.config.cross_attention_dim}, got {combined_cross.shape[-1]}"

                        # First prepare the latents
                        debug_print("\n=== Debug: Before prepare_latents ===")
                        debug_print(f"VAE device: {pipe.vae.device}")
                        debug_print(f"Image device: {image.device}")
                        debug_print(f"Scheduler exists: {noise_scheduler is not None}")

                        # Prepare latents
                        noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, noise_scheduler)

                        debug_print("\n=== Debug: After prepare_latents ===")
                        debug_print(f"Noisy latents: shape={noisy_latents.shape}, dtype={noisy_latents.dtype}")
                        debug_print(f"Noise: shape={noise.shape}, dtype={noise.dtype}")
                        debug_print(f"Timesteps: shape={timesteps.shape}, dtype={timesteps.dtype}")

                        # Create added conditioning kwargs
                        added_cond_kwargs = {
                            "text_embeds": torch.zeros(image.shape[0], 1280).to(accelerator.device),
                            "time_ids": torch.zeros(image.shape[0], 6).to(accelerator.device)
                        }

                        # Now verify all shapes after everything is created
                        debug_print("\n=== Debug: Final Input Tensor Shapes ===")
                        debug_print(f"noisy_latents: {noisy_latents.shape} (expected: [batch_size, 4, height/8, width/8])")
                        debug_print(f"timesteps: {timesteps.shape} (expected: [batch_size])")
                        debug_print(f"encoder_hidden_states: {combined_cross.shape} (expected: [batch_size, 1, 2048])")
                        debug_print(f"controlnet_cond: {normal_map.shape} (expected: [batch_size, 3, height, width])")
                        for k, v in added_cond_kwargs.items():
                            debug_print(f"{k}: {v.shape}")

                        # Verify all shapes are correct
                        try:
                            assert noisy_latents.shape[1] == 4, f"Expected 4 channels in latents, got {noisy_latents.shape[1]}"
                            assert normal_map.shape[1] == 3, f"Expected 3 channels in normal map, got {normal_map.shape[1]}"
                            assert combined_cross.shape[-1] == cross_attention_dim, \
                                f"Expected hidden size of {cross_attention_dim}, got {combined_cross.shape[-1]}"
                        except AssertionError as e:
                            print(f"❌ Verification failed: {str(e)}")
                            raise

                        # Before ControlNet forward pass
                        debug_print("\n=== Debug: ControlNet State ===")
                        debug_print(f"ControlNet training mode: {pipe.controlnet.training}")
                        debug_print(f"ControlNet device: {pipe.controlnet.device}")

                        # Check input tensors
                        debug_print("\nInput tensor stats:")
                        debug_print(f"normal_map - min: {normal_map.min()}, max: {normal_map.max()}, mean: {normal_map.mean()}")
                        debug_print(f"noisy_latents - min: {noisy_latents.min()}, max: {noisy_latents.max()}, mean: {noisy_latents.mean()}")

                        # Add this before the ControlNet forward pass
                        for param in pipe.controlnet.parameters():
                            if param.requires_grad:
                                param.retain_grad()

                        # ControlNet forward pass
                        try:
                            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=combined_cross,  # Use 2048-dim for ControlNet
                                controlnet_cond=normal_map,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )
                        except Exception as e:
                            print(f"Error in ControlNet forward pass: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                            raise

                        debug_print("\n=== Debug: Post-ControlNet Forward Pass ===")
                        debug_print(f"Forward pass successful")
                        debug_print(f"Number of down block samples: {len(down_block_res_samples)}")
                        for i, sample in enumerate(down_block_res_samples):
                            debug_print(f"- Down block {i} shape: {sample.shape}")
                        debug_print(f"Mid block sample shape: {mid_block_res_sample.shape}")

                        # Before UNet forward pass
                        debug_print("\n=== Debug: Residual Connections ===")
                        debug_print(f"Down block residuals stats:")
                        for i, res in enumerate(down_block_res_samples):
                            debug_print(f"- Block {i}: min={res.min().item():.4f}, max={res.max().item():.4f}, mean={res.mean().item():.4f}")
                        debug_print(f"Mid block residual stats: min={mid_block_res_sample.min().item():.4f}, max={mid_block_res_sample.max().item():.4f}, mean={mid_block_res_sample.mean().item():.4f}")

                        # After the ControlNet forward pass
                        debug_print("\n=== Debug: ControlNet Gradients ===")
                        for name, param in pipe.controlnet.named_parameters():
                            if param.requires_grad:
                                debug_print(f"Before backward - {name}:")
                                debug_print(f"- requires_grad: {param.requires_grad}")
                                debug_print(f"- has_grad: {param.grad is not None}")
                                if param.grad is not None:
                                    debug_print(f"- grad_norm: {param.grad.norm().item()}")

                        # Use same conditioning in UNet
                        try:
                            noise_pred = pipe.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=combined_cross,  # Use 2048-dim for UNet
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs
                            ).sample

                            diffusion_loss = loss_fn(noise_pred, noise)
                            cosine_loss = cosine_loss_fn(combined_cross.squeeze(1), id_cross, torch.ones(id_cross.size(0)).to(accelerator.device))
                            controlnet_loss = sum(res.abs().mean() for res in down_block_res_samples)
                            controlnet_loss += mid_block_res_sample.abs().mean()
                            total_loss = 2 * diffusion_loss + config["training"].get("lambda_identity", 0.5) * cosine_loss + 0.1 * controlnet_loss

                            accelerator.backward(total_loss)

                            # pick one of your lora tensors
                            for n,p in pipe.controlnet.controlnet_cond_embedding.named_parameters():
                                if "lora_A" in n:
                                    print(n, "grad norm =", p.grad.norm().item())
                                    break
                            lr_scheduler.step()
                            optimizer.step()
                        except Exception as e:
                            print(f"Error in UNet forward pass: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                            raise

                        # Update EMA
                        ema.step(pipe.unet.parameters())

                        if global_step % config["training"].get("log_steps", 10) == 0:
                            # Get current learning rate
                            current_lr = optimizer.param_groups[0]["lr"]
                            
                            # Calculate gradients statistics
                            grad_norm_unet = 0.0
                            grad_norm_controlnet = 0.0
                            for name, param in pipe.unet.named_parameters():
                                if param.grad is not None:
                                    grad_norm_unet += param.grad.data.norm(2).item() ** 2
                            for name, param in pipe.controlnet.named_parameters():
                                if param.grad is not None:
                                    grad_norm_controlnet += param.grad.data.norm(2).item() ** 2
                            grad_norm_unet = grad_norm_unet ** 0.5
                            grad_norm_controlnet = grad_norm_controlnet ** 0.5
                            
                            # Log statistics
                            accelerator.print(
                                f"\n=== Training Statistics ===\n"
                                f"Epoch: {epoch}/{config['training']['num_epochs']} | "
                                f"Step: {global_step}/{config['training']['max_train_steps']} | "
                                f"Learning Rate: {current_lr:.6f}\n"
                                f"Losses:\n"
                                f"- Total: {total_loss.item():.4f}\n"
                                f"- Diffusion: {diffusion_loss.item():.4f}\n"
                                f"- Identity: {cosine_loss.item():.4f}\n"
                                f"Gradient Norms:\n"
                                f"- UNet: {grad_norm_unet:.4f}\n"
                                f"- ControlNet: {grad_norm_controlnet:.4f}\n"
                                f"Memory:\n"
                                f"- Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB\n"
                                f"- Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB\n"
                                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )  
                    
                        # Save checkpoints
                        if global_step % config["training"].get("save_steps", 500) == 0:    
                            if accelerator.is_main_process:
                                checkpoint_dir = output_dir / "checkpoints"
                                
                                save_path = checkpoint_dir / f"checkpoint-{epoch}-{global_step}"
                                save_path.mkdir(parents=True, exist_ok=True)
                                print(f"\nSaving checkpoint to: {save_path}")
                                
                                # Save LoRA weights separately
                                unwrapped_unet = accelerator.unwrap_model(pipe.unet)
                                unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
                                
                                # Save only the LoRA state dict instead of using save_pretrained
                                unet_lora_state_dict = unwrapped_unet.state_dict()
                                controlnet_lora_state_dict = unwrapped_controlnet.state_dict()
                                
                                # Save files and verify
                                checkpoint_files = {
                                    "unet_lora.pt": unet_lora_state_dict,
                                    "controlnet_lora.pt": controlnet_lora_state_dict,
                                    "lighting_to_cross.pt": lighting_to_cross.state_dict(),
                                    "lighting_to_pooled.pt": lighting_to_pooled.state_dict(),
                                    "identity_to_cross.pt": identity_to_cross.state_dict(),
                                    "identity_to_pooled.pt": identity_to_pooled.state_dict(),
                                    "training_state.pt": {
                                        "step": step,
                                        "epoch": epoch,
                                        "optimizer": optimizer.state_dict(),
                                                "scheduler": lr_scheduler.state_dict(),
                                        "ema": ema.state_dict(),
                                        "config": config,
                                    }
                                }
                                
                                for filename, data in checkpoint_files.items():
                                    file_path = save_path / filename
                                    torch.save(data, file_path)
                                    print(f"Saved {filename}: {file_path.exists()}")

                                CHARACTER_DATA_VOLUME.commit()
                                print("✅ Committed volume changes")

                        # Add the preview generation code:
                        if global_step % config["training"].get("preview_steps", 100) == 0:
                            accelerator.print("\nGenerating preview image...")
                            
                            try:
                                # Ensure models are in eval mode
                                ema.store(pipe.unet.parameters())
                                ema.copy_to(pipe.unet.parameters())
                                pipe.unet.eval()
                                pipe.controlnet.eval()
                                pipe.vae.eval()
                            
                                device = accelerator.device
                                
                                # Get sample normal map and normalize it to [0, 1] range
                                sample_normal = batch["normal_map"][0:1].to(device, dtype=torch.bfloat16)
                                # Normalize from [-1, 1] to [0, 1]
                                sample_normal = (sample_normal + 1.0) / 2.0
                                # Clamp to ensure we're exactly in [0, 1]
                                sample_normal = torch.clamp(sample_normal, 0.0, 1.0)
                                
                                debug_print(f"\nNormalized normal map stats:")
                                debug_print(f"Min: {sample_normal.min():.4f}")
                                debug_print(f"Max: {sample_normal.max():.4f}")
                                debug_print(f"Mean: {sample_normal.mean():.4f}")
                                debug_print(f"Std: {sample_normal.std():.4f}")
                                
                                # Process embeddings exactly like in training loop
                                raw_id = F.normalize(batch["embedding"][0:1].to(device), dim=-1)
                                raw_light = F.normalize(batch["lighting"][0:1].to(device), dim=-1)

                                # Create embeddings once
                                id_cross = identity_to_cross(raw_id).to(dtype=torch.bfloat16)      # [1,2048]
                                light_cross = lighting_to_cross(raw_light).to(dtype=torch.bfloat16)  # [1,2048]
                                combined_cross = (id_cross + light_cross).unsqueeze(1)               # [1,1,2048]

                                # Before ControlNet forward pass
                                print("\nVerifying dimensions before preview generation:")
                                print(f"combined_cross shape: {combined_cross.shape}")
                                print(f"ControlNet cross attention dim: {pipe.controlnet.config.cross_attention_dim}")
                                
                                # Ensure dimensions match
                                assert combined_cross.shape[-1] == pipe.controlnet.config.cross_attention_dim, \
                                    f"Dimension mismatch: combined_cross ({combined_cross.shape[-1]}) vs ControlNet ({pipe.controlnet.config.cross_attention_dim})"
                                
                                # Use the same embeddings for both models
                                down_samples, mid_sample = pipe.controlnet(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=combined_cross,
                                    controlnet_cond=sample_normal,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )

                                # Use 2048-dim for UNet
                                noise_pred = pipe.unet(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=combined_cross,  # Use 2048-dim for UNet
                                    down_block_additional_residuals=down_samples,
                                    mid_block_additional_residual=mid_sample,
                                    added_cond_kwargs=added_cond_kwargs,
                                ).sample

                                # 5)  Denoise one step (or full 25‑step loop if you like)
                                prev_sample = noise_scheduler.step(noise_pred, timesteps, noisy_latents).prev_sample

                                # 6)  Decode to image
                                decoded = pipe.vae.decode(prev_sample / 0.18215).sample  # [1,3,512,512] float32
                                decoded = (decoded / 2 + 0.5).clamp(0,1)  # scale from [-1,1]→[0,1]

                                # 6)  Save
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                preview_path = output_dir / "previews" / f"preview_e{epoch}_s{global_step}_{timestamp}.png"
                                preview_path.parent.mkdir(parents=True, exist_ok=True)
                                save_image(decoded, str(preview_path))
                                accelerator.print(f"Preview saved to {preview_path}")

                            except Exception as e:
                                debug_print("\n=== Preview Generation Failed ===")
                                debug_print(f"Error type: {type(e).__name__}")
                                debug_print(f"Error message: {str(e)}")
                                import traceback
                                debug_print(f"\nTraceback:\n{traceback.format_exc()}")
                                raise

                        # Log before zeroing gradients
                        optimizer.zero_grad()

                        global_step += 1

                if global_step >= config["training"]["max_train_steps"]:
                    break
                
            # Validation pass
            pipe.unet.eval()
            total_val_loss = 0
            val_diffusion_losses = []
            val_identity_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    image = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
                    normal_map = batch["normal_map"].to(accelerator.device, dtype=torch.bfloat16)
                    identity = F.normalize(batch["embedding"].to(accelerator.device), dim=-1)
                    lighting = F.normalize(batch["lighting"].to(accelerator.device), dim=-1)

                    # Use the existing identity_projection
                    id_cross = identity_to_cross(identity)  # This uses the properly initialized projection
                    light_cross = lighting_to_cross(lighting)
                    combined_cross = (id_cross + light_cross).unsqueeze(1)

                    noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, noise_scheduler)
                    added_cond_kwargs = {
                        "text_embeds": torch.zeros(image.shape[0], 1280).to(accelerator.device),
                        "time_ids": torch.zeros(image.shape[0], 6).to(accelerator.device)
                    }

                    # Before ControlNet forward pass
                    debug_print("\n=== Debug: ControlNet Input Details ===")
                    debug_print(f"ControlNet config:")
                    debug_print(f"- cross_attention_dim: {pipe.controlnet.config.cross_attention_dim}")
                    debug_print(f"- addition_embed_type: {pipe.controlnet.config.addition_embed_type}")
                    debug_print(f"- projection_class_embeddings_input_dim: {pipe.controlnet.config.projection_class_embeddings_input_dim}")
                    
                    debug_print("\nInput tensor details:")
                    debug_print(f"noisy_latents - shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}, device: {noisy_latents.device}")
                    debug_print(f"timesteps - shape: {timesteps.shape}, dtype: {timesteps.dtype}, device: {timesteps.device}")
                    debug_print(f"encoder_hidden_states - shape: {combined_cross.shape}, dtype: {combined_cross.dtype}, device: {combined_cross.device}")
                    debug_print(f"controlnet_cond - shape: {normal_map.shape}, dtype: {normal_map.dtype}, device: {normal_map.device}")
                    debug_print("\nAdded conditioning kwargs:")
                    for k, v in added_cond_kwargs.items():
                        debug_print(f"- {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

                    # ControlNet forward pass
                    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=combined_cross,
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
                            encoder_hidden_states=combined_cross,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample
                    except Exception as e:
                        print(f"Error in unet: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        raise

                    try:
                        diffusion_loss = loss_fn(noise_pred, noise)
                        cosine_loss = cosine_loss_fn(combined_cross.squeeze(1), id_cross, torch.ones(id_cross.size(0)).to(accelerator.device))
                        val_loss = diffusion_loss + config["training"].get("lambda_identity", 0.5) * cosine_loss
                    except Exception as e:
                        print(f"Error in loss calculation: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        raise
                    
                    val_diffusion_losses.append(diffusion_loss.item())
                    val_identity_losses.append(cosine_loss.item())
                    total_val_loss += val_loss.item()

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
                    ema.store(pipe.unet.parameters())
                    best_model_path = output_dir / "best_model"
                    best_model_path.mkdir(parents=True, exist_ok=True)
                    
                    # Unwrap models
                    unwrapped_unet = accelerator.unwrap_model(pipe.unet)
                    unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
                    
                    # Save LoRA state dicts directly
                    torch.save(unwrapped_unet.state_dict(), best_model_path / "unet_lora.pt")
                    torch.save(unwrapped_controlnet.state_dict(), best_model_path / "controlnet_lora.pt")
                    
                    # Save other components
                    torch.save(lighting_to_cross.state_dict(), best_model_path / "lighting_to_cross.pt")
                    torch.save(lighting_to_pooled.state_dict(), best_model_path / "lighting_to_pooled.pt")
                    
                    # Save training state
                    training_state = {
                        "best_loss": best_loss,
                        "config": config,
                        "ema": ema.state_dict(),
                        "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"  # Add this for reference
                    }
                    torch.save(training_state, best_model_path / "training_state.pt")

                    CHARACTER_DATA_VOLUME.commit()
                    print("✅ Committed volume changes")
                    
                    ema.restore(pipe.unet.parameters())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        if accelerator.is_main_process:
            final_path = output_dir / "final_model"
            final_path.mkdir(parents=True, exist_ok=True)
            
            # Unwrap and save final LoRA weights
            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
            
            # Save all model components
            torch.save(unwrapped_unet.state_dict(), final_path / "unet_lora.pt")
            torch.save(unwrapped_controlnet.state_dict(), final_path / "controlnet_lora.pt")
            torch.save(lighting_to_cross.state_dict(), final_path / "lighting_to_cross.pt")
            torch.save(lighting_to_pooled.state_dict(), final_path / "lighting_to_pooled.pt")
            torch.save(identity_to_cross.state_dict(), final_path / "identity_to_cross.pt")
            torch.save(identity_to_pooled.state_dict(), final_path / "identity_to_pooled.pt")
            
            # Save final training state
            training_state = {
                "config": config,
                "ema": ema.state_dict(),
                "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0"
            }
            torch.save(training_state, final_path / "training_state.pt")
            CHARACTER_DATA_VOLUME.commit()
            print("✅ Committed volume changes")

        # After optimizer step
        if global_step % 10 == 0:  # Check every 10 steps
            debug_print("\n=== Debug: Parameter Updates ===")
            for name, param in pipe.controlnet.named_parameters():
                if param.requires_grad:
                    debug_print(f"{name}:")
                    debug_print(f"- param_norm: {param.norm().item()}")
                    if param.grad is not None:
                        debug_print(f"- grad_norm: {param.grad.norm().item()}")

        # After backward pass
        debug_print("\n=== Debug: Gradient Flow ===")
        debug_print("ControlNet gradients:")
        for name, param in pipe.controlnet.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                if grad_norm > 0:
                    debug_print(f"- {name}: {grad_norm:.6f}")

        debug_print("\nUNet gradients:")
        for name, param in pipe.unet.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                if grad_norm > 0:
                    debug_print(f"- {name}: {grad_norm:.6f}")

        # Add after backward pass
        for name, param in pipe.controlnet.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"Warning: {name} has no gradient")

        # Add these new debug statements
        debug_print("\nPipeline configuration:")
        debug_print(f"UNet config: {pipe.unet.config}")
        debug_print(f"ControlNet config: {pipe.controlnet.config}")
        debug_print(f"VAE config: {pipe.vae.config}")

        # After applying LoRA to controlnet
        pipe.controlnet = get_peft_model(pipe.controlnet, controlnet_lora_config)
        pipe.controlnet.train()

        # Add verification of dimensions
        print("\nVerifying ControlNet dimensions after LoRA:")
        print(f"cross_attention_dim: {pipe.controlnet.config.cross_attention_dim}")
        print(f"projection_class_embeddings_input_dim: {pipe.controlnet.config.projection_class_embeddings_input_dim}")

        # If dimensions are wrong, fix them
        pipe.controlnet.config.cross_attention_dim = 2048
        pipe.controlnet.config.conditioning_embedding_out_channels = 2048

        # Verify again
        print("\nControlNet dimensions after fix:")
        print(f"cross_attention_dim: {pipe.controlnet.config.cross_attention_dim}")
        print(f"projection_class_embeddings_input_dim: {pipe.controlnet.config.projection_class_embeddings_input_dim}")

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