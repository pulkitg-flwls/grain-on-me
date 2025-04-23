import cv2
import os
import argparse
from pathlib import Path

def overwrite_with_first_channel(input_dir):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Recursively walk through subfolders (face_mask, mouth_mask, etc.)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if Path(filename).suffix.lower() in image_extensions:
                file_path = os.path.join(root, filename)
                img = cv2.imread(file_path)

                if img is None:
                    print(f"❌ Failed to load: {file_path}")
                    continue

                if img.ndim == 3 and img.shape[2] == 3:
                    # Extract first channel (e.g., Blue in BGR)
                    channel_0 = img[:, :, 0]

                    # Overwrite original file with the single-channel image
                    cv2.imwrite(file_path, channel_0)
                    print(f"✅ Overwritten with first channel: {file_path}")
                else:
                    print(f"⚠️ Skipped (not 3-channel): {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overwrite 3-channel images with their first channel.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root folder containing subfolders.")

    args = parser.parse_args()
    overwrite_with_first_channel(args.input_dir)