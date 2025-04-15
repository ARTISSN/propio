import sys
import subprocess
import importlib
import shutil

# Optional: to suppress DeprecationWarning (if it bothers you)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define required packages and versions
required_versions = {
    "torch": "2.1.2",
    "torchvision": "0.16.2",
    "torchaudio": "2.1.2",
    "diffusers": "0.26.3",
    "transformers": "4.38.2",
    "tokenizers": "0.15.2",  # ✅ Actually exists and compatible
    "huggingface_hub": "0.20.3",
    "peft": "0.8.2",
    "accelerate": "0.27.2",
    "xformers": "0.0.23.post1",
    "numpy": "1.24.4",
    "opencv-python": None,
    "facenet-pytorch": None,
    "pytorch-msssim": None,
    "lpips": "0.1.4"
}

# Function to install package if not already matching version
def install_if_needed(package, version=None, extra_flags="--no-deps -q"):
    try:
        if version:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != version:
                raise ImportError(f"Version mismatch: {installed_version} != {version}")
        print(f"✅ {package} already installed.")
    except Exception:
        pkg_str = f"{package}=={version}" if version else package
        print(f"🔄 Installing {pkg_str} ...")
        subprocess.run(
            f"{sys.executable} -m pip install {extra_flags} {pkg_str}",
            shell=True,
            check=True
        )

# Uninstall potentially conflicting RAPIDS packages
subprocess.run("pip uninstall -y pylibcugraph-cu12 rmm-cu12", shell=True)
!pip install pytorch-msssim
subprocess.run(f"{sys.executable} -m pip install facenet-pytorch --no-deps -q", shell=True, check=True)

# Install each package only if needed
for pkg, ver in required_versions.items():
    install_if_needed(pkg, ver)

import os
import gc
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_module
import functools
from PIL import Image
from torch import nn
from facenet_pytorch import InceptionResnetV1
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
#from diffusers.models.attention import CrossAttention
from peft import get_peft_model, LoraConfig, TaskType
from pytorch_msssim import ms_ssim
import lpips

# ---------- CONFIG ----------
BASE_MODEL = "/kaggle/input/stable-diffusion-1-5-fp16/stable-diffusion-1.5-fp16"
CONTROLNET_MODEL = "/kaggle/input/control-v11p-sd15-normalbae"
IMAGE_SIZE = 512

LORA_NAME = "daniel_lora"
LORA_TRIGGER = "danielface"
FACE_DIR = f"/kaggle/input/d/ianbalaguera/{LORA_TRIGGER}"
NORMAL_DIR = f"/kaggle/input/{LORA_TRIGGER}normal"
TARGET_DIR = "/kaggle/input/framenormals/frame_00092_normal.png"
OUTPUT_DIR = f"/kaggle/working/{LORA_NAME}"
# Optional: set to versioned output dir like "daniel_lora_v2"
OUTPUT_VERSION = "v2"  # ← Change this each time you want to version it
if OUTPUT_VERSION:
    OUTPUT_DIR = f"/kaggle/working/{LORA_NAME}_{OUTPUT_VERSION}"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LIGHTING_ENCODER_PATH = os.path.join(OUTPUT_DIR, "lighting_encoder.pt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
LORA_PROMPT = f"photo of <{LORA_TRIGGER}>, realistic lighting, sharp details"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LIGHTING_ENCODER_PATH = "/kaggle/working/lighting_encoder.pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SWAPOUT_PATH = "/kaggle/working/swap.png"

RUN_TRAINING = False  # Set to True to train, False to only run inference
FIND_CHECKPOINTS = False  # Attempt to resume from previous checkpoints if possible
USE_NAMED_ADAPTER = True  # Toggle to True to use named adapter like "face_lora"
ADAPTER_NAME = "face_lora"

# Relevant if RUN_TRAINING = False
LORA_DATA = f"/kaggle/input/firstrun-facelora/{LORA_NAME}_{OUTPUT_VERSION}"
UNET_LORA = os.path.join(LORA_DATA, "unet_lora")
CONTROLNET_LORA = os.path.join(LORA_DATA, "controlnet_lora")
CHECKPOINTS_DIR = os.path.join(LORA_DATA, "checkpoints")
TOKENIZER_DIR = os.path.join(LORA_DATA, "tokenizer")

BATCH_SIZE = 1
ACCUM_STEPS = 2  # simulate batch_size = 2 * BATCH_SIZE
scaler = GradScaler()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
identity_proj = nn.Linear(512, 768).to(DEVICE)

# ---------- HELPER FUNCTIONS ----------

"""def patch_cross_attention(attn: CrossAttention, extra_dim: int = 512):
    inner_dim = attn.to_q.out_features
    input_dim = attn.to_q.in_features + extra_dim  # default: 768 + 512

    attn.to_q = torch.nn.Linear(input_dim, inner_dim, bias=False).to(attn.to_q.weight.device)
    attn.to_k = torch.nn.Linear(input_dim, inner_dim, bias=False).to(attn.to_k.weight.device)
    attn.to_v = torch.nn.Linear(input_dim, inner_dim, bias=False).to(attn.to_v.weight.device)"""

def safe_ms_ssim_loss(pred, target, device="cuda"):
    try:
        with torch.cuda.amp.autocast(enabled=False):
            score = ms_ssim(pred.detach().float().clamp(0, 1), target.detach().float().clamp(0, 1), data_range=1.0, size_average=True)
        if not torch.isfinite(score):
            raise ValueError(f"Non-finite score: {score}")
        return 1 - torch.clamp(score, min=1e-4, max=0.999)
    except Exception as e:
        print(f"⚠️ MS-SSIM fallback: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)

def get_identity_embedding(face_tensor):
    # face_tensor: torch.Tensor shape (3, H, W), normalized [0, 1], aligned face
    face_tensor = face_tensor.to(dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0))  # output shape: (1, 512)
    return emb.squeeze(0)

def identity_cosine_loss(predicted_image, reference_image, facenet_model):
    pred_embedding = get_identity_embedding(predicted_image.squeeze(0))
    ref_embedding = get_identity_embedding(reference_image.squeeze(0))
    return 1 - F.cosine_similarity(pred_embedding.unsqueeze(0), ref_embedding.unsqueeze(0)).mean()

def apply_identity_dropout(face_tensor, all_faces_dir, dropout_prob=0.2):
    import random
    if random.random() < dropout_prob:
        # Randomly select different image from directory
        other_faces = [f for f in os.listdir(all_faces_dir) if f.endswith(('.png', '.jpg'))]
        random_path = os.path.join(all_faces_dir, random.choice(other_faces))
        alt_face = Image.open(random_path).convert("RGB")
        alt_tensor = transforms.ToTensor()(alt_face).to(DEVICE)
        return get_identity_embedding(alt_tensor)
    else:
        return get_identity_embedding(face_tensor)

def safe_copy_dir(src_dir, dst_dir):
    if os.path.exists(src_dir):
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"✅ Copied: {src_path} → {dst_path}")

def randn_tensor(shape, generator=None, device=None, dtype=torch.float16):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)

def preprocess_controlnet_image(img, dtype=torch.float16, device="cuda"):
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(dtype=dtype, device=device)
    return tensor

def add_custom_token(pipeline, token="<danielface>", init_token="man"):
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    if tokenizer.add_tokens(token):
        text_encoder.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            token_id = tokenizer.convert_tokens_to_ids(token)
            init_id = tokenizer.convert_tokens_to_ids(init_token)
            token_embeds = text_encoder.get_input_embeddings()
            token_embeds.weight[token_id] = token_embeds.weight[init_id]
        print(f"Token {token} initialized from {init_token}.")
    else:
        print(f"Token {token} already exists.")

# ---------- LIGHTING ENCODER ----------
class SHLightingEncoder(nn.Module):
    def __init__(self, in_dim=27, out_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.model(x)

# ---------- DATASET ----------
class FaceLoRADataset(Dataset):
    def __init__(self, face_dir, normal_dir, transform=None):
        self.face_dir = face_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.face_images = sorted(os.listdir(face_dir))
        self.normal_images = sorted(os.listdir(normal_dir))

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        face_path = os.path.join(self.face_dir, self.face_images[idx])
        normal_path = os.path.join(self.normal_dir, self.normal_images[idx])
        
        face_image = Image.open(face_path).convert("RGB")
        normal_image = Image.open(normal_path).convert("RGB")
        
        if self.transform:
            face_image = self.transform(face_image)
            normal_image = self.transform(normal_image)
        
        return {
            "image": face_image,
            "normal_map": normal_image
        }

# ---------- INIT MODELS ----------
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL, 
    torch_dtype=torch.float16,
    variant="fp16",  # <- tells it to look for `diffusion_pytorch_model.fp16.safetensors`
    use_safetensors=True  # <- use safetensors instead of .bin
).to(DEVICE)


# Define LoRA config for UNet (within ControlNet)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)

# Define the config for UNet LoRA manually
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["to_q", "to_k", "to_v", "ff.net.0.proj", "ff.net.2"],  # typical attention layers in UNet
    task_type=TaskType.FEATURE_EXTRACTION  # UNet is not a language model
)

# Then wrap the pipeline.unet
pipeline.unet.add_adapter(
    adapter_name=ADAPTER_NAME,
    adapter_config=lora_config
)
pipeline.unet.set_adapters([ADAPTER_NAME])
pipeline.unet.enable_adapters()
#patch_cross_attention_modules(pipeline.unet, extra_dim=512)
#patch_cross_attention_modules(pipeline.controlnet, extra_dim=512)
pipeline.unet.train()
pipeline.unet.enable_gradient_checkpointing()
pipeline.enable_attention_slicing()

#try:
#    pipeline.enable_xformers_memory_efficient_attention()
#    print("✅ Enabled xFormers memory efficient attention")
#except ModuleNotFoundError:
#    print("⚠️ xFormers not found. Skipping memory efficient attention.")
pipeline.scheduler.set_timesteps(15) 

# Load tokenizer and text encoder
if RUN_TRAINING:
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
else:
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=torch.float16)
pipeline.tokenizer = tokenizer
pipeline.text_encoder = text_encoder.to(DEVICE)

# Add a new token
new_token = "<danielface>"
num_added = tokenizer.add_tokens(new_token)
if num_added == 0:
    print(f"Token {new_token} already exists.")
else:
    print(f"Added token: {new_token}")
    pipeline.text_encoder.resize_token_embeddings(len(tokenizer))

# Optimizations
#pipeline.enable_xformers_memory_efficient_attention()
#pipeline.enable_model_cpu_offload()

# ---------- Load VAE for latent loss ----------
vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=torch.float16).to(DEVICE)
vae.eval()

print("✅ Models loaded:")
print("Base model:", BASE_MODEL)
print("ControlNet:", CONTROLNET_MODEL)

# Lighting encoder (skipped in training for now)
lighting_encoder = SHLightingEncoder().to(DEVICE)

# ---------- TRAINING LOOP ----------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor()
])
dataset = FaceLoRADataset(
    face_dir=FACE_DIR,
    normal_dir=NORMAL_DIR,
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

lora_params = [p for n, p in pipeline.unet.named_parameters() if p.requires_grad] +
    [p for n, p in identity_proj.named_parameters()]

# Force all LoRA and ControlNet parameters to float32 for optimizer compatibility
for p in lora_params:
    if p.requires_grad:
        p.data = p.data.to(torch.float32)

# Initialize optimizer with safe settings
optimizer = torch.optim.Adam(
    lora_params,
    lr=1e-4,
    foreach=False
)

# Initialize LPIPS (only once!)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

# Run a dummy step to initialize optimizer state
dummy = torch.tensor(0.0, requires_grad=True, device=DEVICE)
dummy.backward()
optimizer.zero_grad()

# Only run training if activated
if RUN_TRAINING:
    # Determine starting epoch from checkpoint filename (e.g., "unet_epoch_6.pt")
    start_epoch = 0
    
    # Handle resuming and copying inputs into versioned OUTPUT_DIR
    if FIND_CHECKPOINTS:
        existing_checkpoint = None
        for filename in sorted(os.listdir(CHECKPOINTS_DIR)):
            if filename.endswith(".pt"):
                existing_checkpoint = os.path.join(CHECKPOINTS_DIR, filename)
    
        if existing_checkpoint:
            pipeline.unet.load_state_dict(torch.load(existing_checkpoint))
            print(f"✅ Resumed training from checkpoint: {existing_checkpoint}")
    
            # Copy ALL input files into the new output dir
            print("📦 Resumed — copying previous files into current OUTPUT_DIR for consistency...")
    
            safe_copy_dir(TOKENIZER_DIR, os.path.join(OUTPUT_DIR, "tokenizer"))
            safe_copy_dir(UNET_LORA, os.path.join(OUTPUT_DIR, "unet_lora"))
            safe_copy_dir(CONTROLNET_LORA, os.path.join(OUTPUT_DIR, "controlnet_lora"))
            safe_copy_dir(CHECKPOINTS_DIR, os.path.join(OUTPUT_DIR, "checkpoints"))
    
            # Copy lighting encoder (single file)
            lighting_path_in = os.path.join(LORA_DATA, "lighting_encoder.pt")
            lighting_path_out = os.path.join(OUTPUT_DIR, "lighting_encoder.pt")
            if os.path.exists(lighting_path_in):
                shutil.copy2(lighting_path_in, lighting_path_out)
                print(f"✅ Copied lighting encoder: {lighting_path_in} → {lighting_path_out}")
    
            # Optional: parse epoch
            filename = os.path.basename(existing_checkpoint)
            if "epoch_" in filename:
                try:
                    start_epoch = int(filename.split("_epoch_")[1].split(".")[0]) + 1
                except ValueError:
                    start_epoch = 0
                    print("⚠️ Could not parse epoch number from checkpoint.")
    else:
        print("No existing checkpoint detected. Starting training...")
    
    EPOCHS = 10
    for epoch in range(start_epoch, EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()
        for i, batch in enumerate(dataloader):
            image = batch["image"].to(DEVICE).to(torch.float32)
            normal_map = batch["normal_map"].to(DEVICE).to(torch.float16)
    
            # Convert normal_map to PIL image
            normal_np = (normal_map[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            normal_pil = Image.fromarray(normal_np)
            
            # Use the actual prompt embeddings (index 0 in the tuple)
            prompt = [LORA_PROMPT] * BATCH_SIZE
            # Tokenize prompt manually
            input_ids = pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.to(DEVICE)

            # ✅ Only once per batch
            identity_embedding = apply_identity_dropout(image[0], FACE_DIR)  # (512,)
            identity_embedding = identity_proj(identity_embedding)
            identity_embedding = identity_embedding.unsqueeze(0).repeat(BATCH_SIZE, 1)  # (B, 768)
            identity_embedding = identity_embedding.unsqueeze(1)  # shape: (B, 1, 768)
            pipeline.text_encoder = lambda input_ids, **kwargs: type('obj', (object,), {"last_hidden_state": identity_embedding})()
                        
            # --------- Latent prediction via pipeline ---------
            with autocast():
                # Forward pass & prediction
                #with torch.no_grad():
                    # Prepare latents manually
                latents = randn_tensor(
                    (BATCH_SIZE, pipeline.unet.config.in_channels, 64, 64),
                    generator=torch.Generator(device=DEVICE).manual_seed(42),
                    device=DEVICE,
                    dtype=torch.float16
                )
                latents = latents * pipeline.scheduler.init_noise_sigma

                # Prepare random noise for the diffusion process
                noise = randn_tensor(
                    latents.shape,
                    generator=torch.Generator(device=DEVICE).manual_seed(42),
                    device=DEVICE,
                    dtype=torch.float16
                )
                
                # Add noise to latents (handled implicitly by scheduler in real usage, but we need ground truth for loss)
                noisy_latents = latents + noise  # optional, just for conceptual clarity

                # Run diffusion (with LoRA UNet) and get final latents
                for t_orig in pipeline.scheduler.timesteps:
                    t = t_orig if isinstance(t_orig, torch.Tensor) else torch.tensor(t_orig, dtype=torch.float16, device=DEVICE)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        latent_model_input = pipeline.scheduler.scale_model_input(latents, t)

                        # 🔁 ControlNet forward
                        controlnet_output = pipeline.controlnet(
                            sample=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=identity_embedding,
                            controlnet_cond=normal_map,
                            return_dict=True,
                        )
                
                        # Ensure residuals match UNet dtype
                        controlnet_output.down_block_res_samples = [r.to(torch.float16) for r in controlnet_output.down_block_res_samples]
                        controlnet_output.mid_block_res_sample = controlnet_output.mid_block_res_sample.to(torch.float16)
                
                        unet_output = pipeline.unet(
                            sample=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=identity_embedding,
                            down_block_additional_residuals=controlnet_output.down_block_res_samples,
                            mid_block_additional_residual=controlnet_output.mid_block_res_sample,
                        )
                    noise_pred = unet_output.sample

                # Core diffusion noise loss
                noise_loss = F.mse_loss(noise_pred.float(), noise.float())
                
                # Decode to image for perceptual losses (optional, can be skipped on some steps)
                latents_down = F.interpolate(latents, size=(32, 32), mode="bilinear", align_corners=False)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    reconstructed = pipeline.vae.decode(latents_down / pipeline.vae.config.scaling_factor).sample.clamp(0, 1)

                with torch.no_grad():
                    lat_target = vae.encode(image).latent_dist.mode()
                    lat_target = F.interpolate(lat_target, size=(32, 32), mode="bilinear", align_corners=False)
                lat_pred = vae.encode(reconstructed).latent_dist.mode()
                latent_loss = F.mse_loss(lat_pred, lat_target)

                recon_small = F.interpolate(reconstructed, size=(256, 256), mode='bilinear', align_corners=False)
                image_small = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
                recon_small = recon_small.clamp(0, 1)
                image_small = image_small.clamp(0, 1)
                
                # uncomment this if latent loss is causing OOM
                # latent_loss = torch.tensor(0.0, device=DEVICE)
                cosine_loss = 1 - F.cosine_similarity(
                    recon_small.view(BATCH_SIZE, -1),
                    image_small.view(BATCH_SIZE, -1)
                ).mean()
                
                lpips_loss = lpips_loss_fn(
                    (recon_small * 2 - 1).float(),
                    (image_small * 2 - 1).float()
                ).mean()

                identity_loss = identity_cosine_loss(reconstructed, image, facenet)
                msssim_loss = safe_ms_ssim_loss(recon_small, image_small, device=DEVICE)
                
                # ---------- Combine losses ----------
                loss = (
                    1.0 * noise_loss +
                    0.01 * latent_loss +
                    0.2 * cosine_loss +
                    0.4 * lpips_loss +
                    0.75 * identity_loss +
                    0.25 * msssim_loss 
                ) / ACCUM_STEPS

                print(
                    f"Loss: {loss.item():.4f} | "
                    f"noise: {noise_loss.item():.4f} | "
                    f"mse: {latent_loss.item():.4f} | "
                    f"cos: {cosine_loss.item():.4f} | "
                    f"lpips: {lpips_loss.item():.4f} | "
                    f"identity: {identity_loss.item():.4f} | "
                    f"msssim: {msssim_loss.item():.4f}"
                )

                scaler.scale(loss).backward()

                has_grad = False
                for name, param in pipeline.unet.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        has_grad = True
                        break
                print(f"✅ Gradient present: {has_grad}")

                if (i + 1) % ACCUM_STEPS == 0:
                    skip_step = False

                    # ✅ Convert all gradients to float32 BEFORE unscaling
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None and p.grad.dtype == torch.float16:
                                p.grad.data = p.grad.data.to(torch.float32)
                
                    # ✅ Now it's safe to unscale and check for bad grads
                    scaler.unscale_(optimizer)
                
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                skip_step = True
                                break
                        if skip_step:
                            break
                
                    if skip_step:
                        print("⚠️ Invalid gradients detected — skipping optimizer step.")
                        optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                if i % 5 == 0:
                    print(
                        f"Epoch {epoch} | Step {i} | Total Loss: {loss.item():.4f} | "
                        f"MSE: {latent_loss.item():.4f} | Cosine: {cosine_loss.item():.4f} | LPIPS: {lpips_loss.item():.4f}"
                    )
    
        if epoch % 3 == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/unet_epoch_{epoch}.pt"
            torch.save(pipeline.unet.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            
    # ---------- SAVE LoRA ----------

    controlnet.save_pretrained(os.path.join(OUTPUT_DIR, "controlnet_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    # Remove memory-efficient attention processors for saving
    pipeline.unet.set_attn_processor(None)
    pipeline.unet.save_attn_procs(
        os.path.join(OUTPUT_DIR, "unet_lora"),
        adapter_name=ADAPTER_NAME,
        weight_name=f"{ADAPTER_NAME}.safetensors"
    )
    print(f"✅ LoRA adapters saved to {OUTPUT_DIR}")
    torch.save(lighting_encoder.state_dict(), LIGHTING_ENCODER_PATH)
    print("Training complete. LoRA and lighting encoder saved.")

    # ✅ Update inference source to use current run's output
    LORA_DATA = OUTPUT_DIR
else:
    print("✅ Existing training outputs detected. Skipping training.")
    print("Loading saved LoRA weights...")
    pipeline.unet.load_attn_procs(
        os.path.join(LORA_DATA, "unet_lora"),
        weight_name=f"{ADAPTER_NAME}.safetensors",
        adapter_name=ADAPTER_NAME,
        local_files_only=True
    )
    pipeline.unet.set_adapters([ADAPTER_NAME])
    controlnet.load_state_dict(
        load_file(os.path.join(CONTROLNET_LORA, "diffusion_pytorch_model.safetensors")),
        strict=False
    )
    lighting_encoder.load_state_dict(torch.load("/kaggle/input/firstrun-facelora/lighting_encoder.pt"))
    lighting_encoder.eval()

# ---------- INFERENCE (Face Swap) ----------
from diffusers import StableDiffusionPipeline

# Load target normal map
target_normal = Image.open(TARGET_DIR).convert("RGB").resize((512, 512))

# Load SH lighting if available
try:
    target_sh_path = TARGET_DIR.replace("_normal.png", "_sh.npy")
    target_sh = torch.tensor(np.load(target_sh_path), dtype=torch.float16).unsqueeze(0).to(DEVICE)
    lighting_encoder.eval()
    lighting_embed = lighting_encoder(target_sh)
except FileNotFoundError:
    print("No SH lighting provided — skipping lighting conditioning.")
    lighting_embed = None

# Load pipeline with trained weights
inference_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)
inference_pipeline.unet.load_attn_procs(
    os.path.join(LORA_DATA, "unet_lora"),
    weight_name=f"{ADAPTER_NAME}.safetensors",
    adapter_name=ADAPTER_NAME,
    local_files_only=True  # ← guarantees only local file access
)
inference_pipeline.unet.set_adapters(["face_lora"])
inference_pipeline.unet.to(dtype=torch.float16)
inference_pipeline.controlnet.to(dtype=torch.float16)
inference_pipeline.enable_attention_slicing()
generator = torch.manual_seed(42)

# Compare to training data
for idx, data in enumerate(dataloader):
    normal_map = data["normal_map"].to(dtype=torch.float16, device=DEVICE)
    original_face = data["image"]
    
    # Preprocess and get the identity embedding
    identity_face = original_face[0].to(DEVICE)
    identity_embedding = get_identity_embedding(identity_face).unsqueeze(0)
    
    # Project identity to 768 if needed
    identity_proj = nn.Linear(512, 768).to(DEVICE)
    identity_embedding = identity_embedding.to(dtype=torch.float32)
    identity_embed_proj = identity_proj(identity_embedding).unsqueeze(1).to(torch.float16)

    with torch.autocast("cuda"):
        output = inference_pipeline(
            prompt_embeds=identity_embed_proj,
            negative_prompt_embeds=torch.zeros_like(identity_embed_proj),
            image=preprocess_controlnet_image(normal_map[0].cpu(), dtype=torch.float16),
            num_inference_steps=30,
            generator=generator
        )
        predicted = output.images[0]
    
        # Save side-by-side
        combined = Image.new("RGB", (1024, 512))
        combined.paste(transforms.ToPILImage()(original_face[0].cpu()), (0, 0))
        combined.paste(predicted, (512, 0))
        combined.save(f"/kaggle/working/compare_{idx}.png")
    
        original_np = np.array(transforms.ToPILImage()(original_face[0].cpu()))
        generated_np = np.array(predicted)
        
        gray_original = np.mean(original_np, axis=2)
        gray_generated = np.mean(generated_np, axis=2)
    
        print(gray_original.max() - gray_original.min())
        score = ssim(gray_original, gray_generated, data_range=gray_original.max() - gray_original.min())
        print(f"SSIM Score: {score:.4f}")

with torch.autocast("cuda"):
    # Generate output
    gen_output = inference_pipeline(
        prompt_embeds=identity_embed_proj,
        negative_prompt_embeds=torch.zeros_like(identity_embed_proj),
        image=preprocess_controlnet_image(target_normal, dtype=torch.float16),
        num_inference_steps=30,
        generator=generator
    )
    
    # Save output image
    out_img = gen_output.images[0]
    out_img.save(SWAPOUT_PATH)
    print(f"Face swap complete. Output saved as '{SWAPOUT_PATH}'")