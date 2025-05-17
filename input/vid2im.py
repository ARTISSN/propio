import cv2
import os

# === USER CONFIGURABLE ===
video_path = 'convertvids'        # Path to the input video
output_dir = 'frames_output'            # Directory to save extracted frames
frame_prefix = 'f'                  # Prefix for image filenames
image_format = 'jpg'                    # Can be 'jpg', 'png', etc.

# === SETUP ===
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each video in the input directory
for video_file in os.listdir(video_path):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue
        
    # Get video name without extension
    video_name = os.path.splitext(video_file)[0]
    
    # Open video
    video_path_full = os.path.join(video_path, video_file)
    cap = cv2.VideoCapture(video_path_full)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path_full}")
        continue
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
    
        # Build filename with video name as prefix
        filename = f"{video_name}_{frame_prefix}_{frame_count:05d}.{image_format}"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
    
        frame_count += 1
    
    print(f"✅ Done! Extracted {frame_count} frames from '{video_file}' to '{output_dir}'")
    
    cap.release()
