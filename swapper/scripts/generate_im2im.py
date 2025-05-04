#!/usr/bin/env python3
"""
generate_im2im.py

Applies a LoRA (from a training run) to SDXL and runs img2img on a directory of renders.
Intended for use on Modal, but can be run locally for debugging.
"""

import os
from pathlib import Path
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
import argparse
import yaml

def fix_lora_state_dict_keys(lora_state_dict):
    new_state_dict = {}
    for k, v in lora_state_dict.items():
        if k.endswith('.lora_A.weight'):
            new_k = k.replace('.lora_A.weight', '.lora_A.default.weight')
        elif k.endswith('.lora_B.weight'):
            new_k = k.replace('.lora_B.weight', '.lora_B.default.weight')
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def load_lora(pipe, lora_path, lora_cfg, device="cuda"):
    from peft import get_peft_model
    # Wrap U-Net with LoRA
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    # Load LoRA weights
    lora_state_dict = torch.load(lora_path, map_location=device)
    lora_state_dict = fix_lora_state_dict_keys(lora_state_dict)
    print("LoRA params before loading:")
    for n, p in pipe.unet.named_parameters():
        if "lora" in n:
            print(n, p.data.view(-1)[:5])
    pipe.unet.load_state_dict(lora_state_dict, strict=False)
    print("LoRA params after loading:")
    for n, p in pipe.unet.named_parameters():
        if "lora" in n:
            print(n, p.data.view(-1)[:5])
    print("LoRA state dict keys:", list(lora_state_dict.keys())[:10])
    print("UNet state dict keys:", list(pipe.unet.state_dict().keys())[:10])
    return pipe

def main(
    source_character,
    training_run,
    target_character,
    renders_dir,
    output_dir,
    config_path="configs/train_config.yaml",
    prompt="<rrrdaniel>",
    device="cuda"
):
    # Paths
    base_path = Path("/workspace/data/characters")
    lora_dir = base_path / source_character / "trainings" / training_run / "best_model"
    lora_path = lora_dir / "lora_adapter.pt"
    renders_dir = Path(renders_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading LoRA from: {lora_path}")

    # Load config for LoRA params
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    lora_cfg = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=[
            "conv1", "conv2", "conv_shortcut", "to_q", "to_k", "to_v", "proj_in", "proj_out"
        ],
        bias="none",
    )

    # Load SDXL pipeline on CPU first
    sdxl_cache_path = "/workspace/cache/huggingface/sdxl-base-1.0"
    print(f"\nLoading SDXL pipeline from: {sdxl_cache_path}")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        sdxl_cache_path,
        torch_dtype=torch.float32,
    )

    # Apply LoRA before moving to device
    pipe = load_lora(pipe, lora_path, lora_cfg, device="cpu")

    # Now move to device
    pipe = pipe.to(device)
    pipe.unet.eval()
    print(f"\nSuccessfully loaded LoRA weights and set U-Net to eval mode")

    # Process each render
    print(f"\nProcessing renders from: {renders_dir}")
    seed = 42  # Set this to any integer for reproducibility
    for i, img_path in enumerate(sorted(renders_dir.glob("*.png"))):
        img = Image.open(img_path).convert("RGB")
        # Preprocess: SDXL expects [0,1] float, 3 channels, 1024x1024 or 512x512
        img = img.resize((config["resolution"], config["resolution"]), Image.LANCZOS)
        generator = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            result = pipe(
                image=img,
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=12,
                generator=generator,
            )
        out_img = result.images[0]
        out_path = output_dir / f"{img_path.stem}_lora.png"
        out_img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-character", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--target-character", required=True)
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default="configs/train_config.yaml")
    parser.add_argument("--prompt", default="<rrrdaniel>")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.source_character,
        args.training_run,
        args.target_character,
        args.renders_dir,
        args.output_dir,
        args.config_path,
        args.prompt,
        args.device
    )