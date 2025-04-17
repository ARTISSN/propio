import json, torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from utils.embedding_utils import get_face_embedding_from_array

image_path = "your_test_image.jpg"
embedding_path = "embeddings/faces/test.jpg.json"
lora_weights = "models/lora_weights/lora.safetensors"

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("checkpoints/sdxl_base")
pipe.unet.load_attn_procs(lora_weights)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

# Load embedding
embedding = torch.tensor(json.load(open(embedding_path)))

# Prompt override using embedding (e.g. custom conditioning)
prompt = "photo of a person with embedding-driven identity"
output = pipe(prompt=prompt).images[0]
output.save("generated_face_swap.jpg")