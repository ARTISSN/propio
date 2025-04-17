# utils/modal_utils.py
import modal
import sys
import site
from pathlib import Path
import os
import shutil

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
    """Internal function to download models without Modal decoration"""
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        import os
        
        # Set environment variables for the cache
        os.environ["HF_HOME"] = f"{cache_mount_path}/huggingface"
        os.environ["TORCH_HOME"] = f"{cache_mount_path}/torch"
        
        # Create cache directories
        for subdir in ["huggingface", "torch", "models"]:
            os.makedirs(f"{cache_mount_path}/{subdir}", exist_ok=True)
        
        cache_path = f"{cache_mount_path}/huggingface/sdxl-base-1.0"
        if os.path.exists(cache_path):
            print("Using cached SDXL model")
            return
            
        print("Downloading SDXL base model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
            
        print(f"Saving model to cache: {cache_path}")
        pipe.save_pretrained(cache_path)
        
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # After downloading to cache_mount_path
        os.environ["MODAL_CACHE_DIR"] = CACHE_MOUNT
        print(f"Set MODAL_CACHE_DIR to: {CACHE_MOUNT}")
        
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

    print("\nDirectory structure:")
    for root, dirs, files in os.walk("/root/swapper"):
        print(f"\nDirectory: {root}")
        print("  Files:", files)
        print("  Subdirs:", dirs)

@app.function(
    image=image,
    gpu="A10G",
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
        
        print("Data sync completed successfully")
        print("\nVerifying synced files:")
        for item in volume_path.glob('**/*'):
            print(f"  {item.relative_to(volume_path)}")

    except Exception as e:
        print(f"Error syncing data: {str(e)}", file=sys.stderr)
        raise

def _download_face_landmarks(cache_mount_path: str):
    """Download face landmarks model if not already in volume"""
    landmarks_path = os.path.join(cache_mount_path, "models", "shape_predictor_68_face_landmarks.dat")
    models_dir = os.path.join(cache_mount_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nChecking face landmarks model:")
    print(f"- Models directory: {models_dir}")
    print(f"- Directory exists: {os.path.exists(models_dir)}")
    print(f"- Directory permissions: {oct(os.stat(models_dir).st_mode)[-3:]}")
    print(f"- Target path: {landmarks_path}")
    print(f"- File exists: {os.path.exists(landmarks_path)}")
    
    if os.path.exists(landmarks_path):
        try:
            # Verify file is valid (not empty/corrupted)
            size = os.path.getsize(landmarks_path)
            print(f"- Existing file size: {size} bytes")
            if size > 0:
                print("Using cached face landmarks model")
                return landmarks_path
        except OSError as e:
            print(f"Error checking existing file: {e}")
        
        print("Removing existing file...")
        try:
            os.remove(landmarks_path)
        except OSError as e:
            print(f"Error removing file: {e}")
            # If we can't remove it, try a different path
            landmarks_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.new.dat")
    
    print("\nStarting download process...")
    compressed_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat.bz2")
    
    # Download the file
    print(f"Downloading to: {compressed_path}")
    download_cmd = f"wget -O {compressed_path} http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    result = os.system(download_cmd)
    if result != 0:
        raise RuntimeError("Failed to download landmarks model")
    
    # Verify download was successful
    if not os.path.exists(compressed_path):
        raise RuntimeError("Download completed but compressed file not found")
    
    compressed_size = os.path.getsize(compressed_path)
    print(f"Downloaded file size: {compressed_size} bytes")
    if compressed_size == 0:
        raise RuntimeError("Downloaded file is empty")
    
    # Extract using Python's bz2 module instead of command line
    print(f"Extracting file...")
    import bz2
    try:
        with bz2.open(compressed_path, 'rb') as source, open(landmarks_path, 'wb') as dest:
            dest.write(source.read())
        print("Extraction completed")
    except Exception as e:
        raise RuntimeError(f"Failed to extract landmarks model: {str(e)}")
    finally:
        # Clean up compressed file
        try:
            os.remove(compressed_path)
        except OSError:
            pass
    
    # Final verification
    if not os.path.exists(landmarks_path):
        raise RuntimeError(f"Extraction completed but file not found at {landmarks_path}")
    
    final_size = os.path.getsize(landmarks_path)
    print(f"\nModel downloaded and extracted successfully")
    print(f"- File exists: {os.path.exists(landmarks_path)}")
    print(f"- Final file size: {final_size} bytes")
    
    return landmarks_path

@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={
        DATA_MOUNT: CHARACTER_DATA_VOLUME,
        CACHE_MOUNT: CACHE_VOLUME,
    }
)
def train_remote(character_name: str):
    """Train the model remotely on Modal"""
    try:
        print(f"Setting up training environment for {character_name}")
        
        # Download models to cache first
        print("\nEnsuring models are in cache...")
        _download_models_internal(CACHE_MOUNT)
        
        # Add project directory to Python path FIRST
        project_dir = "/root/swapper"
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
            print(f"Added {project_dir} to Python path")
        
        # Import the training module early
        print("\nImporting training module...")
        try:
            from swapper.scripts import train_sdxl_face_swap as training_module
            print("Successfully imported training module")
        except ImportError as e:
            print(f"Failed to import training module: {str(e)}")
            print("\nCurrent Python path:")
            for p in sys.path:
                print(f"  {p}")
            raise
        
        # Rest of setup...
        for subdir in ["huggingface", "torch", "models", "site-packages"]:
            os.makedirs(f"{CACHE_MOUNT}/{subdir}", exist_ok=True)
        
        os.environ["HF_HOME"] = f"{CACHE_MOUNT}/huggingface"
        os.environ["TORCH_HOME"] = f"{CACHE_MOUNT}/torch"
        
        workspace_landmarks = _download_face_landmarks(CACHE_MOUNT)
        
        # 4. Create local models directory and symlink AFTER download
        os.makedirs("models", exist_ok=True)
        relative_landmarks = "models/shape_predictor_68_face_landmarks.dat"

        print(f"\nChecking paths before symlink:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Source path: {workspace_landmarks}")
        print(f"Source exists: {os.path.exists(workspace_landmarks)}")
        print(f"Source is symlink: {os.path.islink(workspace_landmarks)}")
        if os.path.islink(workspace_landmarks):
            print(f"Source points to: {os.readlink(workspace_landmarks)}")

        # Use absolute paths to avoid symlink chains
        abs_source = os.path.abspath(workspace_landmarks)
        abs_target = os.path.abspath(relative_landmarks)

        print(f"\nAbsolute paths:")
        print(f"Source: {abs_source}")
        print(f"Target: {abs_target}")

        if os.path.exists(abs_target):
            if os.path.islink(abs_target):
                print(f"Removing existing symlink: {abs_target}")
                os.remove(abs_target)
            else:
                print(f"Moving existing file to backup: {abs_target}")
                os.rename(abs_target, f"{abs_target}.backup")

        try:
            # Create a hard copy instead of a symlink
            print(f"Copying file instead of creating symlink")
            shutil.copy2(abs_source, abs_target)
            print(f"File copied successfully")
            print(f"Target exists: {os.path.exists(abs_target)}")
            print(f"Target size: {os.path.getsize(abs_target)} bytes")
            
            # Set environment variable for dlib to find the model
            os.environ["DLIB_SHAPE_PREDICTOR"] = abs_target
            print(f"Set DLIB_SHAPE_PREDICTOR to: {abs_target}")
        except Exception as e:
            print(f"Error copying file: {str(e)}")
            raise
        
        # Print mount points and volumes
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
        return training_module.train(character_name)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {"status": "error", "message": error_msg}

@app.local_entrypoint()
def run_modal_train(character_name: str):
    """Main entry point for training"""
    try:
        print(f"Current working directory: {os.getcwd()}")
        
        # First verify local data exists
        local_char_path = verify_local_data(character_name)
        
        # Create a zip file of the character data
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for item in local_char_path.glob('**/*'):
                if item.is_file():
                    arcname = item.relative_to(local_char_path)
                    zip_file.write(item, arcname)
        
        with app.run():
            print(f"Starting Modal sync for character: {character_name}")
            sync_character_data.remote(character_name, zip_buffer.getvalue())
            
            print(f"Starting Modal training for character: {character_name}")
            result = train_remote.remote(character_name)
            print(f"Training result: {result}")
            return result
            
    except Exception as e:
        error_msg = f"Modal execution failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {"status": "error", "message": error_msg}