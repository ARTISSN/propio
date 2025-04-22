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
from lpips import LPIPS

from swapper.utils.image_utils import preprocess_image
from swapper.utils.embedding_utils import get_face_embedding
from swapper.models.projectors import EmbeddingProjector

from models.lighting_mlp import LightingMLP

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
def load_checkpoint(checkpoint_path: Path, pipe, projector, optimizer, lr_scheduler, ema):
    """Load model and training state from checkpoint."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    try:
        # Load model weights
        unet_path = checkpoint_path / "unet_lora.pt"
        controlnet_path = checkpoint_path / "controlnet_lora.pt"
        projector_path = checkpoint_path / "projector.pt"
        training_state_path = checkpoint_path / "training_state.pt"
        
        required_files = [
            unet_path, controlnet_path, projector_path, training_state_path
        ]
        
        if not all(p.exists() for p in required_files):
            raise FileNotFoundError("Checkpoint files are incomplete")
            
        # Load model weights
        pipe.unet.load_state_dict(torch.load(unet_path))
        pipe.controlnet.load_state_dict(torch.load(controlnet_path))
        projector.load_state_dict(torch.load(projector_path))
        
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
            # 1) Load the base SDXL pipeline
            base = StableDiffusionXLPipeline.from_pretrained(
                sdxl_cache_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
            ).to(accelerator.device)

            # 2) Clone UNet into ControlNet without manual dimension modifications
            controlnet = ControlNetModel.from_unet(base.unet)
            #controlnet.config.addition_embed_type = None

            # 5) Create the pipeline with the properly configured controlnet
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
            print(f"Error loading SDXL model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # Add this before creating the LoRA config
        print("\nAvailable modules in ControlNet:")
        for name, _ in pipe.controlnet.named_modules():
            print(f"- {name}")

        # First, create the LoRA config with the correct module names
        try:
            controlnet_lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    "controlnet_cond_embedding.conv_in",
                    "controlnet_cond_embedding.conv_out",
                    # (uncomment the next lines only if you need to tune cross‑attn)
                    # "controlnet_down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_q",
                    # "controlnet_down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_k",
                    # "controlnet_down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_q",
                    # "controlnet_down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k",
                    # "controlnet_mid_block.attentions.*.transformer_blocks.*.attn1.to_q",
                    # "controlnet_mid_block.attentions.*.transformer_blocks.*.attn1.to_k",
                    # "controlnet_mid_block.attentions.*.transformer_blocks.*.attn2.to_q",
                    # "controlnet_mid_block.attentions.*.transformer_blocks.*.attn2.to_k",
                ],
                init_lora_weights="gaussian",
                bias="none",
                task_type="CONTROLNET"
            )
        except Exception as e:
            print(f"Error creating LoRA config: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

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
        pipe.controlnet.eval()
        for p in pipe.controlnet.parameters():
            p.requires_grad = False

        # Then, use this corrected fix_attention_dimensions function
        def fix_attention_dimensions(module):
            if hasattr(module, "to_k") or hasattr(module, "to_q") or hasattr(module, "to_v"):
                # All attention layers should operate at 1280 dimensions
                output_dim = 1280
                num_heads = 20
                head_dim = output_dim // num_heads
                
                if "attn2" in module.__class__.__name__.lower():
                    # Cross attention
                    input_dim = 2048  # encoder hidden states dimension
                else:
                    # Self attention - always use 1280
                    input_dim = 1280
                
                # Update all projection layers
                if hasattr(module, "to_q"):
                    module.to_q = torch.nn.Linear(input_dim, output_dim, bias=False)
                if hasattr(module, "to_k"):
                    module.to_k = torch.nn.Linear(input_dim, output_dim, bias=False)
                if hasattr(module, "to_v"):
                    module.to_v = torch.nn.Linear(input_dim, output_dim, bias=False)
                
                # Output projection is always to output_dim
                if hasattr(module, "to_out"):
                    module.to_out = torch.nn.ModuleList([
                        torch.nn.Linear(output_dim, output_dim, bias=False),
                        torch.nn.Dropout(p=0.0)
                    ])
                
                # Update attention parameters
                module.heads = num_heads
                module.head_dim = head_dim
                module.scale = head_dim ** -0.5

        # Add this verification step after applying the fix
        def verify_attention_block(name, module):
            if hasattr(module, "to_q"):
                q_in = module.to_q.in_features
                q_out = module.to_q.out_features
                if q_out != 1280:
                    raise ValueError(f"Incorrect output dimension in {name}.to_q: {q_out} (should be 1280)")
                
                # For cross attention, input should be 2048, otherwise 1280
                expected_in = 2048 if "attn2" in name else 1280
                if q_in != expected_in:
                    raise ValueError(f"Incorrect input dimension in {name}.to_q: {q_in} (should be {expected_in})")
                
        # Verify the dimensions
        for model_name, model in [("ControlNet", pipe.controlnet), ("UNet", pipe.unet)]:
            verify_attention_block(model_name, model)

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

        # Initialize projector with correct dimensions
        projector = EmbeddingProjector(
            config=config,
            unet_dim=pipe.unet.config.cross_attention_dim,  # 2048
            controlnet_dim=pipe.controlnet.config.cross_attention_dim  # 2048
        )

        # Move to device and prepare with accelerator
        projector = projector.to(accelerator.device)
        projector = accelerator.prepare(projector)

        # Update optimizer to use only UNet and projector parameters
        optimizer = torch.optim.Adam(
            list(pipe.unet.parameters()) +
            list(projector.parameters()),  # Remove ControlNet parameters
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
                projector,
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
        pipe.unet, pipe.controlnet, projector, \
        optimizer, train_loader, val_loader = accelerator.prepare(
            pipe.unet, pipe.controlnet, projector,
            optimizer, train_loader, val_loader
        )

        # Training loop
        best_loss = float('inf')
        patience = config["training"].get("patience", 5)
        patience_counter = 0

        # Near the beginning of your train function, after creating the accelerator
        lpips_model = LPIPS(net='vgg').to(accelerator.device)

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

                        # For UNet and ControlNet only
                        combined_cross = projector(identity, lighting, target="unet")  # For UNet
                        controlnet_embeds = projector(identity, lighting, target="controlnet")  # For ControlNet

                        # First prepare the latents
                        debug_print("\n=== Debug: Before prepare_latents ===")
                        debug_print(f"VAE device: {pipe.vae.device}")
                        debug_print(f"Image device: {image.device}")
                        debug_print(f"Scheduler exists: {noise_scheduler is not None}")

                        # Prepare latents
                        noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, noise_scheduler)

                        noisy_latents = noisy_latents.requires_grad_(True)
                        noise = noise.detach()  # We don't need gradients for the target noise

                        debug_print("\n=== Debug: After prepare_latents ===")
                        debug_print(f"Noisy latents: shape={noisy_latents.shape}, dtype={noisy_latents.dtype}")
                        debug_print(f"Noise: shape={noise.shape}, dtype={noise.dtype}")
                        debug_print(f"Timesteps: shape={timesteps.shape}, dtype={timesteps.dtype}")

                        # Now verify all shapes after everything is created
                        debug_print("\n=== Debug: Final Input Tensor Shapes ===")
                        debug_print(f"noisy_latents: {noisy_latents.shape} (expected: [batch_size, 4, height/8, width/8])")
                        debug_print(f"timesteps: {timesteps.shape} (expected: [batch_size])")
                        debug_print(f"encoder_hidden_states: {combined_cross.shape} (expected: [batch_size, 1, 2048])")
                        debug_print(f"controlnet_cond: {normal_map.shape} (expected: [batch_size, 3, height, width])")

                        debug_print("\nInput tensor details:")
                        debug_print(f"noisy_latents - shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}, device: {noisy_latents.device}")
                        debug_print(f"timesteps - shape: {timesteps.shape}, dtype: {timesteps.dtype}, device: {timesteps.device}")
                        debug_print(f"encoder_hidden_states - shape: {controlnet_embeds.shape}, dtype: {controlnet_embeds.dtype}, device: {controlnet_embeds.device}")
                        debug_print(f"controlnet_cond - shape: {normal_map.shape}, dtype: {normal_map.dtype}, device: {normal_map.device}")

                        # Inside the training loop, add this before the ControlNet forward pass
                        try:
                            with torch.autograd.detect_anomaly():
                                # Track inputs for gradient flow
                                try:
                                    debug_print("\n=== Input Gradient Tracking ===")
                                    debug_print(f"noisy_latents requires_grad: {noisy_latents.requires_grad}")
                                    debug_print(f"timesteps requires_grad: {timesteps.requires_grad}")
                                    debug_print(f"encoder_hidden_states requires_grad: {controlnet_embeds.requires_grad}")
                                    debug_print(f"normal_map requires_grad: {normal_map.requires_grad}")
                                except Exception as e:
                                    debug_print(f"Error tracking input gradients: {str(e)}")
                                              
                                # Prepare the added_cond_kwargs with proper dimensions
                                added_cond_kwargs = {
                                    "text_embeds": torch.zeros(controlnet_embeds.size(0), 1280, device=controlnet_embeds.device),
                                    "time_ids": torch.zeros(controlnet_embeds.size(0), 6, device=controlnet_embeds.device)
                                }

                                # Always run ControlNet in inference mode
                                with torch.no_grad():
                                    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                                        noisy_latents,
                                        timesteps,
                                        encoder_hidden_states=controlnet_embeds,
                                        controlnet_cond=normal_map,
                                        added_cond_kwargs=added_cond_kwargs,  # Add this parameter
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

                        # Use same conditioning in UNet
                        try:
                            # Create the same added_cond_kwargs that was used for ControlNet
                            added_cond_kwargs = {
                                "text_embeds": torch.zeros(combined_cross.size(0), 1280, device=combined_cross.device),
                                "time_ids": torch.zeros(combined_cross.size(0), 6, device=combined_cross.device)
                            }

                            noise_pred = pipe.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=combined_cross,  # Use 2048-dim for UNet
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs,  # Add this parameter
                            ).sample

                            # Modify loss computation to ensure gradients are maintained
                            diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
                            a = combined_cross
                            b = controlnet_embeds

                            if a.ndim == 3:
                                a = a.squeeze(1)
                            if b.ndim == 3:
                                b = b.squeeze(1)

                            assert a.shape == b.shape, f"Mismatched projector shapes: {a.shape} vs {b.shape}"

                            cosine_loss = cosine_loss_fn(
                                a.float(),
                                b.float(),
                                torch.ones(a.size(0), device=a.device)
                            )
                            controlnet_loss = sum(res.abs().mean() for res in down_block_res_samples)
                            controlnet_loss += mid_block_res_sample.abs().mean()

                            # Use LPIPS for perceptual similarity
                            lpips_loss = F.mse_loss(noise_pred, noise)  # Just use MSE in latent space
                            
                            # Extract lighting features from generated and target images
                            # You can use a simple MLP for this
                            #lighting_pred = lighting_extractor(decoded)
                            #lighting_target = batch["lighting"]
                            
                            #lighting_loss = F.mse_loss(lighting_pred, lighting_target)
                            
                            # Using a pre-trained facial landmark detector
                            #landmark_loss = landmark_detector.get_landmark_loss(decoded, image)
                            
                            # Use a pre-trained normal map estimator to predict normals from generated image
                            #predicted_normal = normal_estimator(decoded)
                            #normal_loss = F.mse_loss(predicted_normal, normal_map)
                            
                            total_loss = (
                                1.0 * diffusion_loss +       # Standard diffusion objective
                                0.5 * cosine_loss +          # Identity preservation
                                0.3 * lpips_loss            # Perceptual similarity
                                #0.3 * lighting_loss +        # Lighting consistency
                                #0.2 * landmark_loss +        # Facial structure preservation
                                #0.3 * normal_loss            # Normal map consistency
                            )

                            # After loss calculation, add this
                            try:
                                debug_print("\n=== Loss Gradient Flow ===")
                                debug_print(f"diffusion_loss: {diffusion_loss.item():.6f}, requires_grad: {diffusion_loss.requires_grad}")
                                debug_print(f"cosine_loss: {cosine_loss.item():.6f}, requires_grad: {cosine_loss.requires_grad}")
                                debug_print(f"controlnet_loss: {controlnet_loss.item():.6f}, requires_grad: {controlnet_loss.requires_grad}")
                                debug_print(f"total_loss: {total_loss.item():.6f}, requires_grad: {total_loss.requires_grad}")
                                debug_print(f"total_loss.grad_fn: {total_loss.grad_fn}")
                            except Exception as e:
                                debug_print(f"Error in loss gradient flow tracking: {str(e)}")

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
                                f"- Total: {total_loss.item():.4f}\n"
                                f"- Diffusion: {diffusion_loss.item():.4f}\n"
                                f"- Identity: {cosine_loss.item():.4f}\n"
                                f"Gradient Norms:\n"
                                f"- UNet: {grad_norm_unet:.4f}\n"
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
                                    "projector.pt": projector.state_dict(),
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
                                # 1) Put everything in eval
                                ema.store(pipe.unet.parameters())
                                ema.copy_to(pipe.unet.parameters())
                                pipe.unet.eval()
                                pipe.controlnet.eval()
                                pipe.vae.eval()

                                device = accelerator.device

                                # 2) Prepare normal map in [0,1], float32
                                sample_normal = batch["normal_map"][0:1].to(device)
                                sample_normal = ((sample_normal + 1.0) / 2.0).clamp(0, 1).to(torch.float32)

                                # 3) Build your embeddings
                                raw_id = F.normalize(batch["embedding"][0:1].to(device), dim=-1)
                                raw_light = F.normalize(batch["lighting"][0:1].to(device), dim=-1)

                                unet_emb = projector(raw_id, raw_light, target="unet")
                                ctrl_emb = projector(raw_id, raw_light, target="controlnet")

                                # 4) Prepare latents & timesteps
                                noisy_latents, noise, timesteps = prepare_latents(
                                    pipe.vae,
                                    batch["pixel_values"][0:1].to(device, dtype=pipe.vae.dtype),
                                    noise_scheduler
                                )
                                noisy_latents = noisy_latents.requires_grad_(False)

                                # In the preview generation code
                                added_cond_kwargs = {
                                    "text_embeds": torch.zeros(ctrl_emb.size(0), 1280, device=ctrl_emb.device),
                                    "time_ids": torch.zeros(ctrl_emb.size(0), 6, device=ctrl_emb.device)
                                }

                                with torch.no_grad():
                                    # 6) ControlNet forward
                                    down_samples, mid_sample = pipe.controlnet(
                                        noisy_latents,
                                        timesteps,
                                        encoder_hidden_states=ctrl_emb,
                                        controlnet_cond=sample_normal,
                                        added_cond_kwargs=added_cond_kwargs,
                                        return_dict=False,
                                    )

                                    # 7) UNet forward
                                    noise_pred = pipe.unet(
                                        noisy_latents,
                                        timesteps,
                                        encoder_hidden_states=unet_emb,
                                        down_block_additional_residuals=down_samples,
                                        mid_block_additional_residual=mid_sample,
                                        added_cond_kwargs=added_cond_kwargs,  # Add this parameter
                                    ).sample

                                    # 8) One-step denoise + decode
                                    # Move timesteps to CPU to match the scheduler's internal tensors
                                    cpu_timesteps = timesteps.cpu()
                                    
                                    # Run scheduler with CPU tensors
                                    latents = noise_scheduler.step(
                                        noise_pred.detach().cpu(), 
                                        cpu_timesteps, 
                                        noisy_latents.detach().cpu()
                                    ).prev_sample
                                    
                                    # Move results back to GPU for the rest of processing
                                    latents = latents.to(device)
                                    
                                    # Decode to pixel space
                                    decoded_images = pipe.vae.decode(latents / 0.18215).sample
                                    # Scale to [0,1] range
                                    decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

                                # Create a side-by-side comparison
                                comparison = torch.cat([
                                    sample_normal,  # Original normal map
                                    decoded_images,  # Generated image
                                ], dim=3)  # Concatenate horizontally

                                # Save the comparison
                                preview_path = output_dir / "previews" / f"preview_e{epoch}_s{global_step}.png"
                                preview_path.parent.mkdir(exist_ok=True, parents=True)
                                save_image(comparison, str(preview_path))
                                accelerator.print(f"✅ Preview saved to {preview_path}")
                                
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

                    # Use the existing projector
                    combined_cross = projector(identity, lighting, target="unet")  # For UNet
                    controlnet_embeds = projector(identity, lighting, target="controlnet")  # For ControlNet

                    noisy_latents, noise, timesteps = prepare_latents(pipe.vae, image, noise_scheduler)
                    
                    debug_print("\nInput tensor details:")
                    debug_print(f"noisy_latents - shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}, device: {noisy_latents.device}")
                    debug_print(f"timesteps - shape: {timesteps.shape}, dtype: {timesteps.dtype}, device: {timesteps.device}")
                    debug_print(f"encoder_hidden_states - shape: {combined_cross.shape}, dtype: {combined_cross.dtype}, device: {combined_cross.device}")
                    debug_print(f"controlnet_cond - shape: {normal_map.shape}, dtype: {normal_map.dtype}, device: {normal_map.device}")

                    # In the validation loop
                    added_cond_kwargs = {
                        "text_embeds": torch.zeros(controlnet_embeds.size(0), 1280, device=controlnet_embeds.device),
                        "time_ids": torch.zeros(controlnet_embeds.size(0), 6, device=controlnet_embeds.device)
                    }

                    # ControlNet forward pass
                    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=controlnet_embeds,
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
                        # Create the same added_cond_kwargs that was used for ControlNet
                        added_cond_kwargs = {
                            "text_embeds": torch.zeros(combined_cross.size(0), 1280, device=combined_cross.device),
                            "time_ids": torch.zeros(combined_cross.size(0), 6, device=combined_cross.device)
                        }

                        noise_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=combined_cross,  # Use 2048-dim for UNet
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
                        diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
                        a = combined_cross
                        b = controlnet_embeds

                        if a.ndim == 3:
                            a = a.squeeze(1)
                        if b.ndim == 3:
                            b = b.squeeze(1)

                        assert a.shape == b.shape, f"Mismatched projector shapes: {a.shape} vs {b.shape}"

                        cosine_loss = cosine_loss_fn(
                            a.float(),
                            b.float(),
                            torch.ones(a.size(0), device=a.device)
                        )
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
                    torch.save(projector.state_dict(), best_model_path / "projector.pt")
                    
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
            torch.save(projector.state_dict(), final_path / "projector.pt")
            
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

        debug_print(pipe.controlnet.base_model.controlnet_cond_embedding.conv_in.lora_A.default.weight.grad.norm())

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

def save_checkpoint(checkpoint_path: Path, pipe, projector, optimizer, lr_scheduler, ema, step, epoch, config):
    """Save model and training state to checkpoint."""
    try:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(pipe.unet.state_dict(), checkpoint_path / "unet_lora.pt")
        torch.save(projector.state_dict(), checkpoint_path / "controlnet_lora.pt")
        
        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "config": config,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "ema": ema.state_dict()
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        print("Checkpoint saved successfully")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())