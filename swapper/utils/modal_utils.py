# utils/modal_utils.py
import modal
import sys
import site
from pathlib import Path
import os
import shutil
import json
from typing import Optional, Dict
import datetime

MODAL_GPU = "A100"  # More VRAM than A10G

# Create app for Modal
app = modal.App("face-swap-training")

# Define mount paths
MOUNT_ROOT = "/workspace"
DATA_MOUNT = f"{MOUNT_ROOT}/data/characters"
CACHE_MOUNT = f"{MOUNT_ROOT}/cache"  # Single mount point for all cached/persistent data

# Define volumes for persistent storage
CHARACTER_DATA_VOLUME = modal.Volume.from_name("character-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)  # Single volume for all caches

REQUIREMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "modal_requirements.txt")

def verify_local_data(character_name: str):
    """Verify character data exists locally before starting Modal"""
    local_char_path = Path(os.getcwd()) / "data" / "characters" / character_name
    print(f"Checking for character data in: {local_char_path}")
    if not local_char_path.exists():
        raise ValueError(f"Character directory not found locally: {local_char_path}")
    return local_char_path

# Create image with requirements
image = (modal.Image.debian_slim(python_version="3.10")
         # 1. System dependencies first
         .apt_install([
             "git",
             "wget", 
             "curl",
             "libgl1-mesa-glx",
             "libglib2.0-0",
             "build-essential",
             "bzip2"
         ])
         # 2. Python dependencies
         .pip_install(
             "torch==2.1.1+cu118", 
             "torchvision==0.16.1+cu118",
             extra_index_url="https://download.pytorch.org/whl/cu118"
         )
         .pip_install("huggingface-hub==0.19.4")
         .pip_install_from_requirements(REQUIREMENTS_PATH)
         # 3. Add local files
         .add_local_dir(".", remote_path="/root/swapper"))

def _download_models_internal(cache_mount_path: str):
    """Internal function to download SDXL model without Modal decoration"""
    try:
        import torch
        # alias any missing quant-dtypes
        for name in range(1, 9):
            name = "uint"+str(name)
            if not hasattr(torch, name):
                # pick closest real dtype: bool for 1-bit, uint8 for the rest
                setattr(torch, name, torch.bool if name=="uint1" else torch.uint8)
        from diffusers import StableDiffusionXLPipeline
        import os
        
        # Set environment variables for the cache
        os.environ["HF_HOME"] = f"{cache_mount_path}/huggingface"
        os.environ["TORCH_HOME"] = f"{cache_mount_path}/torch"
        
        # Create cache directories
        for subdir in ["huggingface", "torch"]:
            os.makedirs(f"{cache_mount_path}/{subdir}", exist_ok=True)
        
        cache_path = f"{cache_mount_path}/huggingface/sdxl-base-1.0"
        if os.path.exists(cache_path):
            print("Using cached SDXL model")
            return
            
        print("Downloading SDXL base model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
        )
            
        print(f"Saving model to cache: {cache_path}")
        pipe.save_pretrained(cache_path)
        
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        os.environ["MODAL_CACHE_DIR"] = cache_mount_path
        print(f"Set MODAL_CACHE_DIR to: {cache_mount_path}")
        
    except Exception as e:
        print(f"Error in download_models: {str(e)}", file=sys.stderr)
        raise

def debug_python_env():
    """Debug helper to print Python environment details"""
    import sys
    import site
    import pkg_resources

    print("\n=== Python Environment Debug ===")
    print(f"Python version: {sys.version}")
    print(f"\nPYTHONPATH:")
    for p in sys.path:
        print(f"  {p}")
    
    print(f"\nSite packages:")
    for p in site.getsitepackages():
        print(f"  {p}")
    
    print(f"\nInstalled packages:")
    for pkg in pkg_resources.working_set:
        print(f"  {pkg.key} - Version: {pkg.version}")

    #print("\nDirectory structure:")
    #for root, dirs, files in os.walk("/root/swapper"):
    #    print(f"\nDirectory: {root}")
    #    print("  Files:", files)
    #    print("  Subdirs:", dirs)
    
@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=86400,
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
        CACHE_MOUNT: CACHE_VOLUME,
    }
)
def sync_character_data(character_name: str, data_bytes: bytes):
    """Sync local character data to Modal volume"""
    try:
        volume_path = Path(DATA_MOUNT) / character_name
        volume_path.mkdir(parents=True, exist_ok=True)

        # Write the data to a temporary file
        temp_zip = volume_path / "temp.zip"
        temp_zip.write_bytes(data_bytes)

        # Extract the zip file
        import zipfile
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(volume_path)
        
        # Remove the temporary zip file
        temp_zip.unlink()
        
        # Fix metadata paths and validate embeddings
        metadata_path = volume_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update paths in metadata to use forward slashes and correct paths
            if "frames" in metadata:
                for frame_data in metadata["frames"].values():
                    if "maps" in frame_data:
                        for key in ["face", "normal"]:
                            if key in frame_data["maps"]:
                                # Normalize path
                                path = frame_data["maps"][key].replace("\\", "/")
                                # Remove duplicate character directory if present
                                path = path.replace(f"characters/{character_name}/", "", 1)
                                frame_data["maps"][key] = path
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print("Data sync completed successfully")
        #print("\nVerifying synced files:")
        #for item in volume_path.glob('**/*'):
        #    print(f"  {item.relative_to(volume_path)}")

    except Exception as e:
        print(f"Error syncing data: {str(e)}", file=sys.stderr)
        raise

def get_training_dir_name(training_name: Optional[str] = None) -> str:
    """Generate a name for the training directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if training_name:
        # Sanitize custom name and append timestamp
        safe_name = "".join(c if c.isalnum() else "_" for c in training_name)
        return f"training_{safe_name}_{timestamp}"
    return f"training_{timestamp}"

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=86400,
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
        CACHE_MOUNT: CACHE_VOLUME,
    }
)
def train_remote(character_name: str, training_config: Optional[Dict] = None):
    """Train the model remotely on Modal with support for checkpoints and multiple runs."""
    try:
        # Verify Modal volume mount points
        print("\nVerifying Modal volume mounts:")
        print(f"DATA_MOUNT path: {DATA_MOUNT}")
        print(f"CACHE_MOUNT path: {CACHE_MOUNT}")

        # Set up paths
        character_path = Path(DATA_MOUNT) / character_name
        trainings_dir = character_path / "trainings"
        trainings_dir.mkdir(exist_ok=True)
        
        # Check if volumes are mounted and writable
        data_mount = Path(DATA_MOUNT)
        cache_mount = Path(CACHE_MOUNT)
        
        print(f"Data mount exists: {data_mount.exists()}")
        print(f"Cache mount exists: {cache_mount.exists()}")
        
        # Test write access
        test_file = data_mount / "test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("✓ Data volume is writable")
        except Exception as e:
            print(f"! Warning: Data volume write test failed: {e}")
        
        print(f"Setting up training environment for {character_name}")
        
        # Handle training configuration
        training_config = training_config or {}
        from_checkpoint = training_config.get("from_checkpoint")
        if from_checkpoint:
            # Verify checkpoint exists
            checkpoint_dir = trainings_dir / from_checkpoint
            if not checkpoint_dir.exists():
                raise ValueError(f"Checkpoint directory not found: {from_checkpoint}")
            
            # Create new training directory
            training_dir = trainings_dir / get_training_dir_name(training_config.get("training_name"))
            
            # Copy checkpoint contents to new training directory
            print(f"Copying checkpoint from {checkpoint_dir} to {training_dir}")
            shutil.copytree(checkpoint_dir, training_dir, dirs_exist_ok=True)
        else:
            # Create new training directory
            training_dir = trainings_dir / get_training_dir_name(training_config.get("training_name"))
            training_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTraining will be saved to: {training_dir}")
        
        # Download SDXL model to cache first
        print("\nEnsuring SDXL model is in cache...")
        _download_models_internal(CACHE_MOUNT)
        
        # Add project directory to Python path FIRST
        project_dir = "/root/swapper"
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
            print(f"Added {project_dir} to Python path")

        # Create subdirectories
        for subdir in ["checkpoints", "previews", "best_model"]:
            (training_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Import the training module early
        print("\nImporting training module...")
        try:
            from swapper.scripts import train_im2im as training_module
            print("Successfully imported training module")
        except ImportError as e:
            print(f"Failed to import training module: {str(e)}")
            print("\nCurrent Python path:")
            for p in sys.path:
                print(f"  {p}")
            raise
        
        # Set up cache directories
        for subdir in ["huggingface", "torch", "site-packages"]:
            os.makedirs(f"{CACHE_MOUNT}/{subdir}", exist_ok=True)
        
        # Set environment variables for model caches
        os.environ["HF_HOME"] = f"{CACHE_MOUNT}/huggingface"
        os.environ["TORCH_HOME"] = f"{CACHE_MOUNT}/torch"
        os.environ["MODAL_CACHE_DIR"] = CACHE_MOUNT

        # Set the DLIB_SHAPE_PREDICTOR environment variable
        dlib_shape_predictor_path = os.path.join(CACHE_MOUNT, "models", "shape_predictor_68_face_landmarks.dat")
        if os.path.exists(dlib_shape_predictor_path):
            os.environ["DLIB_SHAPE_PREDICTOR"] = dlib_shape_predictor_path
            print(f"Set DLIB_SHAPE_PREDICTOR to: {os.environ['DLIB_SHAPE_PREDICTOR']}")
        else:
            print(f"WARNING: DLIB_SHAPE_PREDICTOR path does not exist: {dlib_shape_predictor_path}")
        
        # Print mount points and volumes for debugging
        print("\nDebug volume mounts:")
        print(f"Current directory: {os.getcwd()}")
        os.system("df -h")
        print("\nDirectory structure:")
        os.system("ls -R /workspace")
        
        # Debug environment
        print("\nEnvironment variables:")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Install package in development mode
        os.chdir(project_dir)
        os.system("pip install -e .")
        
        # Run debug helper
        debug_python_env()
        
        # Create symlinks to cache pip and Python packages
        os.makedirs(f"{CACHE_MOUNT}/site-packages", exist_ok=True)
        
        # Import site here to ensure it's in scope
        import site
        site_packages = next(p for p in site.getsitepackages() if 'site-packages' in p)
        if not os.path.exists(site_packages):
            os.symlink(f"{CACHE_MOUNT}/site-packages", site_packages)
        
        # Now use the already imported module
        print("Starting training...")
        # Update the output directory in the training function
        result = training_module.train_lora(
            character_name=character_name,
            output_dir=str(training_dir),
            from_checkpoint=bool(from_checkpoint)
        )
    
        if result["status"] == "success":
            # Create a symlink to the latest training
            latest_link = character_path / "latest_training"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(training_dir.relative_to(character_path))
            
            print(f"✓ Created symlink to latest training: {latest_link}")
                
            return result
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {"status": "error", "message": error_msg}

@app.function(
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
    }
)
def verify_saved_files(character_name: str):
    """Verify that files were saved correctly to Modal volume"""
    output_dir = Path(DATA_MOUNT) / character_name / "lora_output"
    
    print(f"\nVerifying saved files in: {output_dir}")
    
    if not output_dir.exists():
        print("! Error: Output directory does not exist")
        return False
        
    # Check directory structure
    dirs_to_check = ["checkpoints", "previews", "best_model"]
    for dir_name in dirs_to_check:
        dir_path = output_dir / dir_name
        print(f"\nChecking {dir_name} directory: {dir_path}")
        print(f"Exists: {dir_path.exists()}")
        if dir_path.exists():
            files = list(dir_path.glob("**/*"))
            print(f"Contains {len(files)} files:")
            for f in files:
                if f.is_file():
                    print(f"- {f.relative_to(dir_path)} ({f.stat().st_size / 1024:.1f}KB)")
    
    return True

@app.local_entrypoint()
def run_modal_train(character_name: str, training_config: Optional[Dict] = None):
    """Run the training pipeline with support for checkpoints and multiple runs."""
    try:
        with app.run():
            # First verify local data exists
            local_char_path = verify_local_data(character_name)
            
            print(f"\nStarting training pipeline for {character_name}")
            if training_config and training_config.get("from_checkpoint"):
                print(f"Resuming from checkpoint: {training_config['from_checkpoint']}")
            
            # Sync data and train
            sync_character_data.remote(character_name, create_character_zip(local_char_path))
            
            print("Data sync completed, starting training...")
            result = train_remote.remote(character_name, training_config)
            
            if result["status"] == "success":
                print("\nTraining completed successfully")
                verify_saved_files.remote(character_name)
                
                # Download model
                #print("\nDownloading trained model...")
                #local_model_dir = Path("data/characters") / character_name / "trained_model"
                #downloaded_files = download_trained_model.remote(character_name, str(local_model_dir))
                
                #for name, content in downloaded_files.items():
                #    file_path = local_model_dir / name
                #    file_path.write_bytes(content)
                #    print(f"Saved {name} to {file_path}")

            return result
                
    except Exception as e:
        if "APP_STATE_STOPPED" in str(e) or "heartbeat" in str(e).lower():
            print(f"\nModal connection interrupted (possibly due to sleep/network).")
        raise

@app.function(
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
    }
)
def download_trained_model(character_name: str, local_path: str):
    """Download trained model files to local directory"""
    try:
        output_dir = Path(DATA_MOUNT) / character_name / "lora_output"
        local_dir = Path(local_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model = output_dir / "best_model"
        if not best_model.exists():
            raise ValueError(f"No best model found for {character_name}")
            
        # Read files and return their contents
        model_files = {
            "unet_lora.pt": best_model / "unet_lora.pt",
            "controlnet_lora.pt": best_model / "controlnet_lora.pt",
            "lighting_mlp.pt": best_model / "lighting_mlp.pt",
            "training_state.pt": best_model / "training_state.pt"
        }
        
        downloaded_files = {}
        for name, path in model_files.items():
            if path.exists():
                with open(path, 'rb') as f:
                    downloaded_files[name] = f.read()
                    
        return downloaded_files
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {str(e)}")

@app.function(
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
    }
)
def list_training_artifacts(character_name: str):
    """List all training artifacts saved for a character"""
    try:
        output_dir = Path(DATA_MOUNT) / character_name / "lora_output"
        print(f"\nListing training artifacts for {character_name}")
        print(f"Output directory: {output_dir}")
        
        if not output_dir.exists():
            print("No training artifacts found")
            return
            
        print("\nDirectory structure:")
        for item in output_dir.rglob("*"):
            if item.is_file():
                print(f"- {item.relative_to(output_dir)} ({item.stat().st_size / 1024:.1f}KB)")
                
        # Check specific important files
        checkpoints = list((output_dir / "checkpoints").glob("checkpoint-*"))
        if checkpoints:
            print(f"\nFound {len(checkpoints)} checkpoints:")
            for cp in checkpoints:
                print(f"- {cp.name}")
                
        best_model = output_dir / "best_model"
        if best_model.exists():
            print("\nBest model files:")
            for f in best_model.glob("*"):
                print(f"- {f.name} ({f.stat().st_size / 1024:.1f}KB)")
                
    except Exception as e:
        print(f"Error listing artifacts: {str(e)}")

def create_character_zip(character_path: Path) -> bytes:
    """Create a zip file of character data and return it as bytes"""
    import io
    import zipfile
    
    print(f"Creating zip file from: {character_path}")
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through all files in the character directory
        for item in character_path.glob('**/*'):
            if item.is_file():
                # Get the relative path for the archive
                arcname = item.relative_to(character_path)
                #print(f"Adding file: {arcname}")
                zip_file.write(item, arcname)
    
    return zip_buffer.getvalue()

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=86400,
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
        CACHE_MOUNT: CACHE_VOLUME,
    }
)
def run_modal_generate_im2im(
    source_character: str,
    training_run: str,
    target_character: str,
    renders_dir: str,
    output_dir: str,
    config_path: str = "configs/train_config.yaml",
    prompt: str = "<rrrdaniel>"
):
    print("RUN MODAL GEN")
    import sys
    sys.path.insert(0, "/root/swapper")
    from swapper.scripts.generate_im2im import main as generate_main
    generate_main(
        source_character,
        training_run,
        target_character,
        to_modal_path(renders_dir),
        to_modal_path(output_dir),
        config_path="/root/swapper/configs/train_config.yaml",
        prompt=prompt,
        device="cuda"
    )
    fetch_swapped_faces(target_character, Path(output_dir))

def get_file_manifest(directory: Path):
    manifest = {}
    for file in directory.rglob("*"):
        if file.is_file():
            manifest[str(file.relative_to(directory))] = os.path.getmtime(file)
    return manifest

def load_manifest(manifest_path: Path):
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict, manifest_path: Path):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

def create_incremental_zip(directory: Path, last_manifest: dict):
    import io, zipfile
    new_manifest = get_file_manifest(directory)
    changed_files = [
        f for f, mtime in new_manifest.items()
        if f not in last_manifest or last_manifest[f] != mtime
    ]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for rel_path in changed_files:
            zip_file.write(directory / rel_path, rel_path)
    return zip_buffer.getvalue(), new_manifest, changed_files

def sync_character_incremental(character_name: str, char_path: Path):
    manifest_path = char_path / ".last_synced_manifest.json"
    last_manifest = load_manifest(manifest_path)
    zip_bytes, new_manifest, changed_files = create_incremental_zip(char_path, last_manifest)
    if changed_files:
        print(f"Syncing {len(changed_files)} changed/new files for {character_name}")
        sync_character_data.remote(character_name, zip_bytes)
        save_manifest(new_manifest, manifest_path)
    else:
        print(f"No changes to sync for {character_name}")

def get_deleted_files(last_manifest, new_manifest):
    return [f for f in last_manifest if f not in new_manifest]

def to_modal_path(local_path: str) -> str:
    # Convert to string and normalize slashes
    local_path = str(local_path).replace("\\", "/")
    # Remove leading './'
    if local_path.startswith("./"):
        local_path = local_path[2:]
    # Remove leading 'data/' if present
    if local_path.startswith("data/"):
        local_path = local_path[5:]
    # Remove any leading slash
    if local_path.startswith("/"):
        local_path = local_path[1:]
    return f"/workspace/data/{local_path}"

@app.function(
    image=image,
    gpu=None,  # No GPU needed for file operations
    timeout=600,
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
    }
)
def download_swapped_faces(character_name: str):
    """
    Zips and returns the swapped faces directory for a character.
    """
    import io
    import zipfile
    from pathlib import Path

    #swapped_dir = Path(DATA_MOUNT) / character_name / "processed" / "swapped"
    swapped_dir = "tmp"/ Path(DATA_MOUNT) / character_name / "processed" / "swapped"
    zip_buffer = io.BytesIO()
    if not swapped_dir.exists():
        raise FileNotFoundError(f"No swapped directory found for {character_name}")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in swapped_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(swapped_dir)
                zip_file.write(file, arcname)
    return zip_buffer.getvalue()

def fetch_swapped_faces(character_name: str, local_target_dir: Path):
    """
    Downloads swapped faces from Modal and extracts them to local_target_dir.
    """

    print(f"Requesting swapped faces for {character_name} from Modal...")
    zip_bytes = download_swapped_faces.remote(character_name)
    local_target_dir.mkdir(parents=True, exist_ok=True)
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_file:
        zip_file.extractall(local_target_dir)
    print(f"Swapped faces downloaded to {local_target_dir}")