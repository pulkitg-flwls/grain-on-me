import cv2
import numpy as np
import matplotlib.pyplot as plt
from scatter_grain import ScatterGrain
import os
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count

def process_single_image(args):
    """
    Demonstrates the complete scatter grain workflow with visualization
    Optimized implementation using vectorized operations and library functions.
    
    Parameters:
    -----------
    noisy_path : str
        Path to original noisy plate
    denoised_path : str
        Path to denoised version of original
    edited_path : str
        Path to edited/composited plate
    mask_path : str
        Path to mask indicating edited regions
    output_path : str
        Path to save final output
    """
    noisy_path, denoised_path, edited_path, mask_path, output_path = args
    # Load input images
    noisy = cv2.imread(noisy_path)
    denoised = cv2.imread(denoised_path)
    edited = cv2.imread(edited_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize to 0-1 range - vectorized operation
    noisy = noisy.astype(np.float32) / 255.0
    denoised = denoised.astype(np.float32) / 255.0
    edited = edited.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    
    # Convert to RGB for visualization
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    edited_rgb = cv2.cvtColor(edited, cv2.COLOR_BGR2RGB)
    
    # Initialize ScatterGrain
    
    sg = ScatterGrain()
    
    
    result = sg.apply_matched_grain(noisy, denoised, edited, mask, blend_mode='add')
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imwrite(output_path, result_uint8)
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run example scatter grain workflow")
    parser.add_argument("--noisy", required=True, help="Path to noisy image")
    parser.add_argument("--denoised", required=True, help="Path to denoised image")
    parser.add_argument("--edited", required=True, help="Path to edited image")
    parser.add_argument("--mask", required=True, help="Path to edit mask")
    parser.add_argument("--output", required=True, help="Path to save output image")
    
    args = parser.parse_args()

    os.makedirs(args.output,exist_ok=True)
    img_pairs = []
    for i,imgs in enumerate(tqdm(sorted(os.listdir(args.noisy)))):
        grainy_path = os.path.join(args.noisy,imgs)
        denoise_path = os.path.join(args.denoised,imgs)
        edited_path = os.path.join(args.edited,imgs)
        # mask_path = args.mask
        mask_path = os.path.join(args.mask,imgs)
        output_path = os.path.join(args.output,imgs)
        img_pairs.append((grainy_path,denoise_path,edited_path,mask_path,output_path))
    
    # for srgb vmdf02_ep01_pt03_0050
    # for i,imgs in enumerate(tqdm(range(0,120))):
    #     grainy_path = os.path.join(args.noisy,f"{imgs+1:06d}.jpg")
    #     denoise_path = os.path.join(args.denoised,f"{imgs:06d}.jpg")
    #     edited_path = os.path.join(args.edited,f"{imgs:06d}.jpg")
    #     # mask_path = args.mask
    #     mask_path = os.path.join(args.mask,f"{imgs:06d}.jpg")
    #     output_path = os.path.join(args.output,f"{imgs:06d}.jpg")
    #     img_pairs.append((grainy_path,denoise_path,edited_path,mask_path,output_path))
        
    num_workers = min(cpu_count(),len(img_pairs))
    # num_workers = 1
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)