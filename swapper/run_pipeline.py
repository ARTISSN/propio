# scripts/run_pipeline.py
import os
import json
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import mediapipe as mp
import datetime
import utils.modal_utils as modal_utils
from utils.mesh_utils import generate_face_mesh, generate_normal_maps
from utils.lighting_utils import LightingProcessor
from utils.embedding_utils import get_face_embedding
import numpy as np
from utils.keep_alive import prevent_sleep, allow_sleep
import torch

class CharacterPipeline:
    def __init__(self, character_name: str, base_path: str = "data"):
        self.character_name = character_name
        self.base_path = Path(base_path)
        self.char_path = self.base_path / "characters" / character_name
        self.metadata_path = self.char_path / "metadata.json"
        self.load_metadata()

    def load_metadata(self):
        """Load or create metadata for the character."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "character_name": self.character_name,
                "source_info": {
                    "videos": [],
                    "images": []
                },
                "frames": {
                    # "frame_id": {
                    #     "source_image": "path/to/source.jpg",
                    #     "mesh": {
                    #         "path": "path/to/mesh.obj",
                    #         "generated_at": "timestamp"
                    #     },
                    #     "maps": {
                    #         "face": "path/to/face.png",
                    #         "normal": "path/to/normal.png",
                    #         "generated_at": "timestamp"
                    #     },
                    #     "lighting": {
                    #         "coefficients": [...],
                    #         "generated_at": "timestamp"
                    #     },
                    #     "processing_steps": [
                    #         {
                    #             "type": "mesh_generation",
                    #             "timestamp": "2024-03-20T12:00:00"
                    #         },
                    #         {
                    #             "type": "map_generation",
                    #             "timestamp": "2024-03-20T12:01:00"
                    #         }
                    #     ]
                    # }
                }
            }
            self.save_metadata()

    def save_metadata(self):
        """Save current metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def generate_mesh(self):
        """Generate 3D mesh and normal maps from processed images."""
        print(f"Generating mesh for {self.character_name}")
        
        # Set up input and output directories
        source_image_dir = self.char_path / "source/images"
        processed_dir = self.char_path / "processed"
        mesh_dir = processed_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_image_dir.exists() or not any(source_image_dir.iterdir()):
            print(f"No source images found for {self.character_name}")
            return
            
        # Initialize face mesh
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            print(f"Processing images in: {source_image_dir}")
            # Process each image in the source directory
            for img_path in source_image_dir.glob("*.*"):
                print(f"Processing image: {img_path}")
                try:
                    mesh_data = generate_face_mesh(str(img_path), str(mesh_dir), face_mesh)
                    
                    if mesh_data:
                        base_name = img_path.stem
                        timestamp = datetime.datetime.now().isoformat()
                        
                        # Update metadata with new structure
                        if base_name not in self.metadata["frames"]:
                            self.metadata["frames"][base_name] = {
                                "source_image": str(Path(img_path).relative_to(self.base_path)),
                                "processing_steps": []
                            }
                        
                        self.metadata["frames"][base_name]["mesh"] = {
                            "path": str(Path(mesh_data['obj_path']).relative_to(self.base_path)),
                            "generated_at": timestamp
                        }
                        
                        self.metadata["frames"][base_name]["processing_steps"].append({
                            "type": "mesh_generation",
                            "timestamp": timestamp
                        })
                        
                except Exception as e:
                    print(f"Error processing mesh for {img_path}: {str(e)}")
                    continue
        
        # Save updated metadata
        self.save_metadata()
        print(f"Mesh generation completed for {self.character_name}")

    def generate_maps(self):
        """Generate texture and normal maps, calculate lighting coefficients, and generate face embeddings."""
        print(f"Generating maps, lighting, and embeddings for {self.character_name}")
        
        # Set up directories
        processed_dir = self.char_path / "processed"
        mesh_dir = processed_dir / "meshes"
        maps_dir = processed_dir / "maps"
        faces_dir = maps_dir / "faces"
        normals_dir = maps_dir / "normals"
        
        # Create directories
        for dir_path in [faces_dir, normals_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize lighting processor
        lighting_processor = LightingProcessor(self.base_path, self.character_name)
        
        # Find all mesh files that need processing
        mesh_files = list(mesh_dir.glob("*.obj"))
        if not mesh_files:
            print("No mesh files found to process")
            return
        
        # Process each frame
        for mesh_path in mesh_files:
            base_name = mesh_path.stem
            try:
                # Find source image
                source_image = None
                source_dir = self.char_path / "source/images"
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_source = source_dir / f"{base_name}{ext}"
                    if potential_source.exists():
                        source_image = potential_source
                        break
                
                if not source_image:
                    print(f"Warning: Could not find source image for mesh {base_name}")
                    continue
                
                # Generate face embedding
                print(f"Generating embedding for {source_image}")
                embedding = get_face_embedding(str(source_image))
                if embedding is None:
                    print(f"Warning: Could not generate embedding for {source_image}")
                    embedding = [0.0] * 128  # Default embedding dimension
                
                # Read the source image
                image = cv2.imread(str(source_image))
                if image is None:
                    print(f"Failed to read image: {source_image}")
                    continue
                
                # Initialize face mesh to get landmarks
                with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                ) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if not results.multi_face_landmarks:
                        print(f"No face detected in: {source_image}")
                        continue
                    landmarks = results.multi_face_landmarks[0]
                
                # Generate maps
                mesh_data = {
                    'obj_path': str(mesh_path),
                    'image': image,
                    'landmarks': landmarks
                }
                
                timestamp = datetime.datetime.now().isoformat()
                
                # Generate normal maps and face cutouts
                map_data = generate_normal_maps(mesh_data, str(maps_dir))
                if not map_data:
                    print(f"Failed to generate maps for frame {base_name}")
                    continue
                
                # Process lighting coefficients
                lighting_data = lighting_processor.process_frame(
                    base_name,
                    Path(map_data['face_path']),
                    Path(map_data['normal_path'])
                )
                
                if not lighting_data:
                    print(f"Failed to calculate lighting for frame {base_name}")
                    continue
                
                # Update metadata
                if base_name not in self.metadata["frames"]:
                    self.metadata["frames"][base_name] = {
                        "source_image": str(source_image.relative_to(self.base_path)),
                        "processing_steps": []
                    }
                
                # Update metadata with new data
                self.metadata["frames"][base_name].update({
                    "maps": {
                        "face": str(Path(map_data['face_path']).relative_to(self.base_path)),
                        "normal": str(Path(map_data['normal_path']).relative_to(self.base_path)),
                        "generated_at": timestamp
                    },
                    "lighting": {
                        "coefficients": lighting_data["lighting_coefficients"],
                        "generated_at": timestamp
                    },
                    "embedding": embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding
                })
                
                # Add processing step to history
                self.metadata["frames"][base_name]["processing_steps"].append({
                    "type": "map_and_lighting_generation",
                    "timestamp": timestamp
                })
                
                # Save metadata after each successful frame processing
                self.save_metadata()
                
                print(f"Successfully processed maps, lighting, and embedding for frame {base_name}")
                
            except Exception as e:
                print(f"Error processing frame {base_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Map and lighting generation completed for {self.character_name}")

    def perform_face_swap(self, source_character: str):
        """Generate face swaps using normal maps from source character."""
        print(f"Performing face swap from {source_character} to {self.character_name}")
        
        # Set up paths
        source_char_path = self.base_path / "characters" / source_character
        source_maps_dir = source_char_path / "processed/maps"
        source_metadata_path = source_char_path / "metadata.json"
        
        output_dir = self.char_path / "swapped"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load source character metadata
        if not source_metadata_path.exists():
            raise ValueError(f"Source character {source_character} metadata not found")
            
        with open(source_metadata_path, 'r') as f:
            source_metadata = json.load(f)
        
        # Load our LoRA model path
        lora_path = self.char_path / "lora_output/best_model"
        if not lora_path.exists():
            raise ValueError(f"LoRA model not found for {self.character_name}")
        
        # Call Modal function to perform the swap
        try:
            result = modal_utils.run_modal_swap(
                target_character=self.character_name,
                source_character=source_character,
                source_metadata=source_metadata
            )
            if isinstance(result, dict) and result.get("status") == "error":
                print(f"Face swap failed: {result.get('message')}")
            else:
                print("Face swap completed successfully")
        except Exception as e:
            print(f"Error during face swap: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run character processing pipeline")
    parser.add_argument("character", help="Character name to process")
    parser.add_argument("--base-path", default="data", help="Base path for data directory")
    
    # Pipeline-specific arguments
    parser.add_argument("--mesh", action="store_true", help="Generate 3D mesh")
    parser.add_argument("--maps", action="store_true", help="Generate texture and normal maps")
    
    # Enhanced training arguments
    training_group = parser.add_argument_group('training')
    training_group.add_argument("--train", action="store_true", help="Train LoRA off generated data")
    training_group.add_argument("--from-checkpoint", help="Resume training from a specific checkpoint directory (e.g., 'training_20240418_123456')")
    training_group.add_argument("--training-name", help="Custom name for this training run (default: timestamp-based)")
    
    parser.add_argument("--swap", help="Perform face swap using normal maps from specified source character")
    parser.add_argument("--all", action="store_true", help="Run all pipeline steps")
    args = parser.parse_args()

    pipeline = CharacterPipeline(args.character, args.base_path)
    
    # If no specific steps are specified, run all steps
    run_all = args.all or not (args.mesh or args.maps or args.train)
    
    if args.mesh or run_all:
        pipeline.generate_mesh()
    if args.maps or run_all:
        pipeline.generate_maps()
    if args.train or run_all:
        try:
            print("Starting LoRA training on Modal...")
            prevent_sleep()
            
            # Pass training configuration to Modal
            training_config = {
                "from_checkpoint": args.from_checkpoint,
                "training_name": args.training_name
            }
            
            result = modal_utils.run_modal_train_im2im(args.character)
            if isinstance(result, dict) and result.get("status") == "error":
                print(f"Training failed: {result.get('message')}")
            else:
                print("Training completed successfully")
        except Exception as e:
            print(f"Error during Modal training: {e}")
        finally:
            allow_sleep()

if __name__ == "__main__":
    try:
        prevent_sleep()
        main()
    finally:
        allow_sleep()