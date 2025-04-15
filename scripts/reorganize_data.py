import os
import shutil
import json
from pathlib import Path

def create_character_structure(base_path, character_name):
    """Create the new directory structure for a character."""
    char_path = Path(base_path) / "characters" / character_name
    
    # Create main directories
    dirs = [
        "source/videos",
        "source/images",
        "processed/faces",
        "processed/meshes",
        "processed/maps"
    ]
    
    for dir_path in dirs:
        (char_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create initial metadata.json
    metadata = {
        "character_name": character_name,
        "source_info": {
            "videos": [],
            "images": []
        },
        "face_embeddings": {},
        "lighting_coefficients": {},
        "mesh_parameters": {},
        "processing_history": [],
        "face_attributes": {},
        "file_mappings": {}
    }
    
    with open(char_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return char_path

def migrate_existing_data(old_path, new_path):
    """Migrate data from old structure to new structure."""
    old_path = Path(old_path)
    new_path = Path(new_path)
    
    # Migrate source images
    if (old_path / "src").exists():
        for img in (old_path / "src").glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy2(img, new_path / "source/images")
    
    # Migrate source videos
    if (old_path / "src").exists():
        for vid in (old_path / "src").glob("*"):
            if vid.suffix.lower() in [".mp4", ".avi", ".mov"]:
                shutil.copy2(vid, new_path / "source/videos")
    
    # Migrate processed data
    if (old_path / "normals").exists():
        for norm in (old_path / "normals").glob("*"):
            shutil.copy2(norm, new_path / "processed/maps")
    
    if (old_path / "lighting").exists():
        for light in (old_path / "lighting").glob("*"):
            shutil.copy2(light, new_path / "processed/maps")

def main():
    base_path = Path("data")
    characters = ["daniel", "documale1"]
    
    # Create new structure
    for char in characters:
        new_path = create_character_structure(base_path, char)
        old_path = base_path / char
        
        if old_path.exists():
            migrate_existing_data(old_path, new_path)
            print(f"Migrated data for {char}")
        else:
            print(f"No existing data found for {char}")

if __name__ == "__main__":
    main() 