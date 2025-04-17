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
from utils.lighting_utils import LightingProcessor, calculate_lighting_coefficients

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
                # Convert old format to new format if needed
                if "mesh_parameters" in self.metadata or "processed_frames" in self.metadata:
                    self._migrate_to_new_format()
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

    def _migrate_to_new_format(self):
        """Migrate old metadata format to new format."""
        new_metadata = {
            "character_name": self.metadata["character_name"],
            "source_info": self.metadata["source_info"],
            "frames": {}
        }

        # Merge existing data into new format
        mesh_params = self.metadata.get("mesh_parameters", {})
        processed = self.metadata.get("processed_frames", {})
        history = self.metadata.get("processing_history", [])

        # Create unified frame entries
        for frame_id in set(list(mesh_params.keys()) + list(processed.keys())):
            frame_data = {"processing_steps": []}
            
            # Add mesh data if exists
            if frame_id in mesh_params:
                mesh_info = mesh_params[frame_id]
                frame_data["source_image"] = mesh_info["source_image"]
                frame_data["mesh"] = {
                    "path": mesh_info["obj_path"],
                    "generated_at": None  # Will be updated from history
                }

            # Add processed data if exists
            if frame_id in processed:
                proc_info = processed[frame_id]
                frame_data["maps"] = {
                    "face": proc_info.get("face_map"),
                    "normal": proc_info.get("normal_map"),
                    "generated_at": proc_info.get("timestamp")
                }
                if "lighting_coefficients" in proc_info:
                    frame_data["lighting"] = {
                        "coefficients": proc_info["lighting_coefficients"],
                        "generated_at": proc_info.get("timestamp")
                    }

            # Add relevant history entries
            for entry in history:
                if entry.get("frame") == frame_id:
                    frame_data["processing_steps"].append({
                        "type": entry["type"],
                        "timestamp": entry["timestamp"]
                    })

            new_metadata["frames"][frame_id] = frame_data

        self.metadata = new_metadata

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
            # Process each image in the source directory
            for img_path in source_image_dir.glob("*.[jp][pn][g]"):
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
        """Generate texture and normal maps, and calculate lighting coefficients."""
        print(f"Generating maps and calculating lighting for {self.character_name}")
        
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
        
        # Check which meshes need map generation by comparing with existing maps
        frames_to_process = {}
        for mesh_path in mesh_files:
            base_name = mesh_path.stem
            face_map = faces_dir / f"{base_name}.png"
            normal_map = normals_dir / f"{base_name}.png"
            
            # If either map is missing, we need to process this frame
            if not face_map.exists() or not normal_map.exists():
                # Find corresponding source image
                expression = base_name.split('_')[0] if '_' in base_name else ''
                source_image = None
                
                # Look for source image in metadata first
                if (expression in self.metadata["frames"] and 
                    base_name in self.metadata["frames"][expression] and 
                    "source_image" in self.metadata["frames"][expression][base_name]):
                    source_path = self.base_path / self.metadata["frames"][expression][base_name]["source_image"]
                    if source_path.exists():
                        source_image = source_path
                
                # If not found in metadata, look in source directory
                if not source_image:
                    source_dir = self.char_path / "source/images"
                    if expression:
                        source_dir = source_dir / expression
                    
                    # Look for matching source image with any supported extension
                    for ext in ['.jpg', '.jpeg', '.png']:
                        potential_source = source_dir / f"{base_name}{ext}"
                        if potential_source.exists():
                            source_image = potential_source
                            break
                
                if source_image:
                    frames_to_process[base_name] = {
                        "mesh_path": mesh_path,
                        "source_image": source_image
                    }
                else:
                    print(f"Warning: Could not find source image for mesh {base_name}")
        
        if not frames_to_process:
            print("No new frames to process for map generation")
            return
        
        print(f"Processing maps for {len(frames_to_process)} frames...")
        
        # Process each frame
        for frame_id, frame_data in frames_to_process.items():
            try:
                # Get paths
                obj_path = frame_data["mesh_path"]
                source_image = frame_data["source_image"]
                
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
                    # Get landmarks
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if not results.multi_face_landmarks:
                        print(f"No face detected in: {source_image}")
                        continue
                    landmarks = results.multi_face_landmarks[0]
                
                # Generate maps
                mesh_data = {
                    'obj_path': str(obj_path),
                    'image': image,
                    'landmarks': landmarks
                }
                
                timestamp = datetime.datetime.now().isoformat()
                
                # Generate normal maps and face cutouts
                map_data = generate_normal_maps(mesh_data, str(maps_dir))
                if not map_data:
                    print(f"Failed to generate maps for frame {frame_id}")
                    continue
                
                # Process lighting coefficients
                lighting_data = lighting_processor.process_frame(
                    frame_id,
                    Path(map_data['face_path']),
                    Path(map_data['normal_path'])
                )
                
                if not lighting_data:
                    print(f"Failed to calculate lighting for frame {frame_id}")
                    continue
            
                # Update metadata
                expression = frame_id.split('_')[0] if '_' in frame_id else ''
                if expression not in self.metadata["frames"]:
                    self.metadata["frames"][expression] = {}
                
                if frame_id not in self.metadata["frames"][expression]:
                    self.metadata["frames"][expression][frame_id] = {
                        "source_image": str(source_image.relative_to(self.base_path)),
                        "processing_steps": []
                    }
                
                # Update metadata with new data
                self.metadata["frames"][frame_id].update({
                    "maps": {
                        "face": str(Path(map_data['face_path']).relative_to(self.base_path)),
                        "normal": str(Path(map_data['normal_path']).relative_to(self.base_path)),
                        "generated_at": timestamp
                    },
                    "lighting": {
                        "coefficients": lighting_data["lighting_coefficients"],
                        "generated_at": timestamp
                    }
                })
                
                # Add processing step to history
                self.metadata["frames"][frame_id]["processing_steps"].append({
                    "type": "map_and_lighting_generation",
                    "timestamp": timestamp
                })
                
                # Save metadata after each successful frame processing
                self.save_metadata()
                
                print(f"Successfully processed maps and lighting for frame {frame_id}")
                
            except Exception as e:
                print(f"Error processing frame {frame_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Map and lighting generation completed for {self.character_name}")

def main():
    parser = argparse.ArgumentParser(description="Run character processing pipeline")
    parser.add_argument("character", help="Character name to process")
    parser.add_argument("--base-path", default="data", help="Base path for data directory")
    
    # Pipeline-specific arguments
    parser.add_argument("--mesh", action="store_true", help="Generate 3D mesh")
    parser.add_argument("--maps", action="store_true", help="Generate texture and normal maps")
    parser.add_argument("--train", action="store_true", help="Train LoRA off generated data")
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
            result = modal_utils.run_modal_train(args.character)
            if isinstance(result, dict) and result.get("status") == "error":
                print(f"Training failed: {result.get('message')}")
            else:
                print("Training completed successfully")
        except Exception as e:
            print(f"Error during Modal training: {e}")

if __name__ == "__main__":
    main()