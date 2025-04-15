# scripts/train_lora.py
import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from utils.embedding_utils import get_face_embedding
from utils.image_utils import preprocess_image
from utils.controlnet_utils import create_controlnet_condition
import yaml

# --------------- Configs ----------------
with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --------------- Dataset ----------------
class FaceSwapDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, embedding_dir, normal_dir, lighting_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.embedding_dir = embedding_dir
        self.normal_dir = normal_dir
        self.lighting_dir = lighting_dir

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = preprocess_image(img_path, target_size=(config["resolution"], config["resolution"]))
        embedding_path = os.path.join(config["embedding_dir"], os.path.basename(img_path) + ".json")
        embedding = torch.tensor(torch.load(embedding_path))
        control = create_controlnet_condition(
            normal_map_path=os.path.join(config["normal_dir"], os.path.basename(img_path)),
            lighting_path=os.path.join(config["lighting_dir"], os.path.basename(img_path).replace(".jpg", ".npy"))
        )
        return {"pixel_values": torch.tensor(img).permute(2,0,1), "embedding": embedding, **control}

    def __len__(self):
        return len(self.image_paths)

# --------------- Training ----------------
def main():
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    
    # Load SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16)
    pipe.enable_gradient_checkpointing()
    
    # Add LoRA to UNet
    lora_config = LoraConfig(r=config["lora_rank"])
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Dataset and loader
    dataset = FaceSwapDataset(config["train_images"], config["embedding_dir"], config["normal_dir"], config["lighting_dir"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)

    # Training loop
    pipe.train()
    for step, batch in enumerate(accelerator.prepare(dataloader)):
        with accelerator.accumulate(pipe.unet):
            outputs = pipe(
                prompt_embeds=batch["embedding"],
                image=batch["pixel_values"],
                controlnet_cond=batch["normal_map"],
                additional_condition=batch.get("lighting")
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if step >= config["max_train_steps"]:
            break

    pipe.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()