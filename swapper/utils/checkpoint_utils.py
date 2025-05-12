#!/usr/bin/env python3
"""
checkpoint_utils.py

Utilities for handling model checkpoints, loading, and saving.
"""

import torch
from pathlib import Path
from peft import get_peft_model_state_dict, get_peft_model


def save_lora_state(
    pipe,
    unet_ema,
    output_path,
    config,
    global_step=None,
    epoch=None,
    best_loss=None,
    is_checkpoint=False,
    is_best=False,
    is_final=False
):
    """
    Save the LoRA weights and training state.
    
    Args:
        pipe: The diffusion pipeline containing the UNet with LoRA adapter
        unet_ema: The EMA model for UNet
        output_path: Directory path to save the checkpoint
        config: Training configuration
        global_step: Current training step
        epoch: Current epoch
        best_loss: Best validation loss achieved so far
        is_checkpoint: Whether this is a regular checkpoint
        is_best: Whether this is the best model so far
        is_final: Whether this is the final model
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights - ensure we're getting the state dict in the right format
    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    torch.save(lora_state_dict, output_path / "lora_adapter.pt")

    # Save training state
    training_state = {
        "config": config,
        "ema": unet_ema.state_dict(),
        "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "dtype": str(pipe.unet.dtype)  # Save the dtype for verification
    }
    if is_checkpoint:
        training_state.update({
            "step": global_step,
            "epoch": epoch,
        })
    if is_best:
        training_state["best_loss"] = best_loss

    torch.save(training_state, output_path / "training_state.pt")


def find_lora_checkpoint(training_dir):
    """
    Returns the path to the best, final, or latest checkpoint directory in order of preference.
    
    Args:
        training_dir: Directory containing checkpoints
        
    Returns:
        Path to the most relevant checkpoint directory
        
    Raises:
        FileNotFoundError: If no checkpoint is found
    """
    training_dir = Path(training_dir)
    # 1. Best model
    best_model = training_dir / "best_model"
    if best_model.exists():
        print(f"Loading best model from {best_model}")
        return best_model

    # 2. Final model
    final_model = training_dir / "final_model"
    if final_model.exists():
        print(f"Loading final model from {final_model}")
        return final_model

    # 3. Most recent checkpoint
    checkpoints_dir = training_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda d: d.stat().st_mtime)
            print(f"Loading latest checkpoint from {latest_checkpoint}")
            return latest_checkpoint

    raise FileNotFoundError(f"No LoRA checkpoint found in {training_dir}")


def load_checkpoint(training_dir, unet, lora_cfg=None, device="cpu"):
    """
    Loads the LoRA checkpoint into the provided U-Net model.
    
    Args:
        training_dir: Directory containing checkpoints
        unet: U-Net model to apply checkpoint to
        lora_cfg: LoRA configuration for adapter (required)
        device: Device to load the model to
        
    Returns:
        Tuple containing:
        - The loaded UNet model with LoRA
        - The training state dictionary
        
    Raises:
        FileNotFoundError: If required checkpoint files are missing
        ValueError: If lora_cfg is not provided or if dtype mismatch is detected
    """
    checkpoint_dir = find_lora_checkpoint(training_dir)
    lora_adapter_path = checkpoint_dir / "lora_adapter.pt"
    state_path = checkpoint_dir / "training_state.pt"

    if lora_cfg is None:
        raise ValueError("lora_cfg must be provided to load LoRA adapter weights.")
    
    # Apply LoRA adapter to UNet
    unet = get_peft_model(unet, lora_cfg)
    
    # Load LoRA weights
    if not lora_adapter_path.exists():
        raise FileNotFoundError(f"No LoRA adapter weights found at {lora_adapter_path}")
    
    print(f"Loading LoRA adapter weights from {lora_adapter_path}")
    lora_state_dict = torch.load(lora_adapter_path, map_location=device)
    unet.load_state_dict(lora_state_dict, strict=False)

    if not state_path.exists():
        raise FileNotFoundError(f"Missing training_state.pt in {checkpoint_dir}")
    
    training_state = torch.load(state_path, map_location=device)
    
    # Verify dtype matches
    if "dtype" in training_state:
        expected_dtype = training_state["dtype"]
        if str(unet.dtype) != expected_dtype:
            print(f"Warning: Model dtype {unet.dtype} doesn't match saved dtype {expected_dtype}")
            print("Converting to expected dtype...")
            unet = unet.to(torch.bfloat16)
    else:
        # Default to bfloat16 if dtype not specified
        unet = unet.to(torch.bfloat16)

    return unet, training_state
