# scripts/train_lora.py

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
import yaml

# --------------- Dataset ----------------
class FaceSwapDataset(torch.utils.data.Dataset):
    def __init__(self, character_path):
        """
        Initialize dataset for a specific character.
        
        Args:
            character_path: Path to character directory containing processed data
        """
        self.processed_dir = Path(character_path) / "processed"
        self.maps_dir = self.processed_dir / "maps"
        
        # Set up paths
        self.face_dir = self.maps_dir / "faces"
        self.normal_dir = self.maps_dir / "normals"
        
        # Get all processed face images
        self.face_paths = list(self.face_dir.glob("*.png"))
        
        if not self.face_paths:
            raise ValueError(f"No processed face images found in {self.face_dir}")

    def __getitem__(self, idx):
        face_path = self.face_paths[idx]
        base_name = face_path.stem
        
        # Get corresponding normal map
        normal_path = self.normal_dir / f"{base_name}.png"
        if not normal_path.exists():
            raise ValueError(f"Missing normal map for {base_name}")
        
        # Load and preprocess images
        face_img = preprocess_image(str(face_path), target_size=(config["resolution"], config["resolution"]))
        normal_map = preprocess_image(str(normal_path), target_size=(config["resolution"], config["resolution"]))
        
        # Create control condition
        control = create_controlnet_condition(normal_map)
        
        return {
            "pixel_values": torch.tensor(face_img).permute(2,0,1),
            "normal_map": torch.tensor(normal_map).permute(2,0,1),
            **control
        }

    def __len__(self):
        return len(self.face_paths)

# --------------- Training ----------------
def main(character_name):
    # Load config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set up character path
    character_path = Path(config["base_dir"]) / character_name
    if not character_path.exists():
        raise ValueError(f"Character directory not found: {character_path}")
    
    # Set up output path for this character
    character_output_dir = Path(config["output_dir"]) / character_name
    character_output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    
    # Load SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["sdxl_model_path"], 
        torch_dtype=torch.bfloat16
    )
    pipe.enable_gradient_checkpointing()
    
    # Add LoRA to UNet
    lora_config = LoraConfig(r=config["lora_rank"])
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Initialize dataset with character path
    dataset = FaceSwapDataset(character_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )

    # Training loop
    # ... rest of training code ...
    
    # Save with character-specific path
    pipe.save_pretrained(character_output_dir)

if __name__ == "__main__":
    main()