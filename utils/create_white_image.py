import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def create_white_images(image_folder: str, output_folder: str):
    # Ensure the input folder exists
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"Error: Folder '{image_folder}' does not exist or is not a directory.")
        return
    
    # Get all images in the folder
    image_files = sorted(image_folder.glob("*.*"))  # Sort to maintain order
    image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
    
    if not image_files:
        print("No valid image files found in the folder.")
        return
    
    first_image_path = image_files[0]
    num_frames = len(image_files)
    
    # Read the first image to get its dimensions
    img = cv2.imread(str(first_image_path))
    if img is None:
        print(f"Error: Could not read image '{first_image_path}'.")
        return
    
    height, width, channels = img.shape
    white_image = np.ones((height, width, dtype=np.uint8) * 255
    
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_frames):
        output_path = os.path.join(output_folder,'face_mask',f"{i:06d}.png")
        cv2.imwrite(str(output_path), white_image)
        output_path = os.path.join(output_folder,'mouth_mask',f"{i:06d}.png")
        cv2.imwrite(str(output_path), white_image)
    
    print(f"{num_frames} white images saved in '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate white images from the first image in a folder.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder where white images will be saved.")
    
    args = parser.parse_args()
    create_white_images(args.image_folder, args.output_folder)

