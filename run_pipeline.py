# scripts/run_pipeline.py
import os
import json
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import mediapipe as mp
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

    def process_source_videos(self):
        """Process source videos to extract frames and face data."""
        video_dir = self.char_path / "source/videos"
        if not video_dir.exists():
            print(f"No videos found for {self.character_name}")
            return

        # TODO: Implement video processing
        # - Extract frames
        # - Detect faces
        # - Generate embeddings
        pass

    def process_source_images(self):
        """Process source images to extract face data."""
        image_dir = self.char_path / "source/images"
        if not image_dir.exists():
            print(f"No images found for {self.character_name}")
            return

        # TODO: Implement image processing
        # - Detect faces
        # - Generate embeddings
        pass

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

        # Process each mesh in the metadata
        for base_name, mesh_info in self.metadata["mesh_parameters"].items():
            try:
                # Reconstruct mesh data
                obj_path = self.base_path / mesh_info["obj_path"]
                source_image = self.char_path / "source/images" / mesh_info["source_image"]
                
                if not obj_path.exists() or not source_image.exists():
                    print(f"Missing files for {base_name}, skipping...")
                    continue
                
                # Read the source image
                image = cv2.imread(str(source_image))
                if image is None:
                    print(f"Failed to read image: {source_image}")
                    continue
                
                # Get landmarks and generate maps
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
                    'obj_path': str(obj_path),
                    'image': image,
                    'landmarks': landmarks
                }
                
                map_data = generate_normal_maps(mesh_data, str(maps_dir))
                
                # Process lighting
                lighting_data = lighting_processor.process_frame(
                    base_name,
                    Path(map_data['face_path']),
                    Path(map_data['normal_path'])
                )
                
                if lighting_data:
                    # Update metadata with all information for this frame
                    frame_data = {
                        "source_image": str(Path(source_image).relative_to(self.base_path)),
                        "mesh": str(Path(obj_path).relative_to(self.base_path)),
                        "face_map": lighting_data["face_map"],
                        "normal_map": lighting_data["normal_map"],
                        "lighting_coefficients": lighting_data["lighting_coefficients"],
                        "timestamp": lighting_data["timestamp"]
                    }
                    
                    self.metadata["processed_frames"][base_name] = frame_data
                    
                    # Add to processing history
                    self.metadata["processing_history"].append({
                        "type": "map_and_lighting_generation",
                        "frame": base_name,
                        "timestamp": lighting_data["timestamp"]
                    })
                
            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                continue
        
        # Save updated metadata
        self.save_metadata()
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
    run_all = args.all or not (args.mesh or args.maps)
    
    if args.mesh or run_all:
        pipeline.generate_mesh()
    if args.maps or run_all:
        pipeline.generate_maps()

if __name__ == "__main__":
    main()