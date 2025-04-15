import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new project directory structure."""
    directories = [
        'configs',
        'data/images',
        'data/normal_maps',
        'data/lighting_coeffs',
        'embeddings/faces',
        'models/lora_weights',
        'scripts',
        'utils',
        'checkpoints/sdxl_base',
        'checkpoints/controlnet_pretrained',
        'misc'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def move_files():
    """Move files to their new locations."""
    # Utility files
    shutil.move('drawing_utils.py', 'utils/drawing_utils.py')
    shutil.move('lighting_utils.py', 'utils/lighting_utils.py')
    shutil.move('coordinate_utils.py', 'utils/coordinate_utils.py')
    shutil.move('normal_map_generator.py', 'utils/normal_map_generator.py')
    
    # Scripts
    shutil.move('facemeshmarker.py', 'scripts/extract_embeddings.py')
    shutil.move('facegenerate.py', 'scripts/inference.py')
    shutil.move('vid2im.py', 'scripts/vid2im.py')
    
    # Data files
    shutil.move('face_model_with_iris.obj', 'misc/face_model_with_iris.obj')
    shutil.move('canonical_face_model.obj', 'misc/canonical_face_model.obj')
    
    # Move image directories
    if os.path.exists('frames_output'):
        shutil.move('frames_output', 'data/images/frames_output')
    if os.path.exists('frames'):
        shutil.move('frames', 'data/images/frames')
    if os.path.exists('morephotosofdaniel'):
        shutil.move('morephotosofdaniel', 'data/images/morephotosofdaniel')
    
    # Move normal maps
    if os.path.exists('normals'):
        shutil.move('normals', 'data/normal_maps/normals')
    if os.path.exists('maps'):
        shutil.move('maps', 'data/normal_maps/maps')
    if os.path.exists('Normal'):
        shutil.move('Normal', 'data/normal_maps/Normal')
    
    # Move lighting data
    if os.path.exists('Ao'):
        shutil.move('Ao', 'data/lighting_coeffs/Ao')
    
    # Move debug images to misc
    debug_files = [
        'debug_face_square.png',
        'debug_ao_map.png',
        'debug_image_normal_map.png',
        'debug_mesh_normal_map.png',
        'sphere_norm.png',
        'hemisphere_debug.png',
        'lighting_map.png',
        'sphere_visualization.png',
        'benchmark_lighting_on_normal_map.png',
        'annotated_image0.png',
        'output.obj'
    ]
    
    for file in debug_files:
        if os.path.exists(file):
            shutil.move(file, f'misc/{file}')
    
    # Move test files
    shutil.move('test_normal_conversion.py', 'misc/test_normal_conversion.py')

def create_requirements():
    """Create requirements.txt file."""
    requirements = [
        'numpy',
        'opencv-python',
        'mediapipe',
        'open3d',
        'trimesh',
        'pyvista',
        'torch',
        'torchvision',
        'diffusers',
        'transformers',
        'accelerate',
        'modal',
        'pyyaml'
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

def create_configs():
    """Create configuration files."""
    # Create train_config.yaml
    train_config = """
model:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  controlnet_model: "lllyasviel/sd-controlnet-canny"
  
training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 1e-4
  mixed_precision: "fp16"
  
data:
  image_size: 512
  normal_map_size: 512
  face_crop_size: 512
  
output:
  save_steps: 100
  save_dir: "checkpoints"
"""
    
    # Create sdxl_lora_config.yaml
    sdxl_config = """
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  
training:
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  warmup_steps: 100
  
inference:
  num_inference_steps: 50
  guidance_scale: 7.5
"""
    
    with open('configs/train_config.yaml', 'w') as f:
        f.write(train_config)
    
    with open('configs/sdxl_lora_config.yaml', 'w') as f:
        f.write(sdxl_config)

def update_readme():
    """Update README.md with new project structure."""
    readme = """# Face Swap SDXL

A project for face swapping using Stable Diffusion XL and ControlNet.

## Project Structure

```
face-swap-sdxl/
├── configs/              # Configuration files
├── data/                 # Dataset and processed data
├── embeddings/           # Face embeddings
├── models/              # Model weights
├── scripts/             # Processing and training scripts
├── utils/               # Utility functions
├── checkpoints/         # Training checkpoints
└── misc/               # Miscellaneous files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Modal:
```bash
modal token new
```

3. Set up your data:
- Place face images in `data/images/`
- Place normal maps in `data/normal_maps/`
- Place lighting coefficients in `data/lighting_coeffs/`

## Usage

1. Extract face embeddings:
```bash
python scripts/extract_embeddings.py
```

2. Train LoRA:
```bash
python scripts/train_lora.py
```

3. Generate images:
```bash
python scripts/inference.py
```

## License

MIT License
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)

def main():
    """Main function to reorganize the project."""
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Moving files to new locations...")
    move_files()
    
    print("Creating requirements.txt...")
    create_requirements()
    
    print("Creating configuration files...")
    create_configs()
    
    print("Updating README.md...")
    update_readme()
    
    print("Project reorganization complete!")

if __name__ == "__main__":
    main() 