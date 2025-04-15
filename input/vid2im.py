import cv2
import os

# === USER CONFIGURABLE ===
video_path = 'Male Enhanced.mp4'        # Path to the input video
output_dir = 'frames_output'            # Directory to save extracted frames
frame_prefix = 'frame'                  # Prefix for image filenames
image_format = 'jpg'                    # Can be 'jpg', 'png', etc.

# === SETUP ===
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        break

    # Build filename and save frame
    filename = f"{frame_prefix}_{frame_count:05d}.{image_format}"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)

    frame_count += 1

print(f"✅ Done! Extracted {frame_count} frames to '{output_dir}'.")

cap.release()
cv2.destroyAllWindows()
