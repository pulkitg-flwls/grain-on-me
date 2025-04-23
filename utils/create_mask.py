import argparse
import cv2
import numpy as np
import os
import gc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def create_edit_mask(args):
    # Load images
    original_path, edited_path, threshold, save_path = args
    original = cv2.imread(original_path)
    edited = cv2.imread(edited_path)

    if original is None or edited is None:
        raise FileNotFoundError("One or both images could not be loaded.")

    if original.shape != edited.shape:
        raise ValueError("Original and edited images must have the same dimensions.")

    # Compute absolute difference
    diff = cv2.absdiff(original, edited)

    # Convert to grayscale (optional but helps reduce noise)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    # norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

    # # Optionally apply Gaussian blur to get a soft mask
    # blur_ksize=11
    # soft_mask = cv2.GaussianBlur(norm_diff, (blur_ksize, blur_ksize), 0)

    # Save the result
    cv2.imwrite(save_path, mask)
    # print(f"Mask saved to {save_path}")

def png2jpg(args):

    img_path,output_path = args
    img = cv2.imread(img_path)
    # img = img/255.0
    # img = np.power(img,1/2.2) * 255.0
    cv2.imwrite(output_path,img.astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY), 90])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask of edited region between two images.")
    parser.add_argument("--original", type=str, help="Path to the original image")
    parser.add_argument("--edited", type=str, help="Path to the edited image")
    parser.add_argument("--threshold", type=int, default=10, help="Pixel difference threshold")
    parser.add_argument("--output", type=str, default="edit_mask.png", help="Path to save the mask")

    args = parser.parse_args()
    os.makedirs(args.output,exist_ok=True)
    img_pairs = []
    for i,imgs in enumerate(sorted(os.listdir(args.original))):
        original_path = os.path.join(args.original,imgs)
        edited_path = os.path.join(args.edited,imgs)
        output_path = os.path.join(args.output,imgs.split('.')[0]+'.jpg')
        img_pairs.append((original_path,edited_path,args.threshold,output_path))
        # img_pairs.append((original_path,output_path))
    
    num_workers = min(cpu_count(),len(img_pairs))
    # num_workers = 1
    with Pool(processes=num_workers) as pool:
        results = pool.map(create_edit_mask,img_pairs)
        # results = pool.map(png2jpg,img_pairs)
    # create_edit_mask(args.original, args.edited, args.threshold, args.output)