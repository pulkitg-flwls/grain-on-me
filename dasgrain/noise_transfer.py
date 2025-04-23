import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage
import argparse
import os
from normalize import compute_noise_map, compute_response_curve,normalize_with_polynomial, fit_response_curve,compute_luminance
from patch_image import get_lowest_variance_patch_chunked, tile_noise_patch
from voronoi import generate_voronoi_pattern, scatter_voronoi_cells
from multiprocessing import Pool, cpu_count

def add_adaptive_noise(edited_image, normalized_noise, fitted_curves, noise_func):
    """Adds adapted noise to the edited image using the response curve in 0-1 range."""
    
    adapted_noise = np.zeros_like(edited_image, dtype=np.float32)
    luminance = compute_luminance(edited_image)  # Extract the luminance
    for ch in range(edited_image.shape[2]):
        intensity = edited_image[:, :, ch]  
        # expected_std = noise_func(intensity, *fitted_curves[ch])  # Noise variance in 0-1 range
        expected_std = noise_func(luminance, *fitted_curves[ch])
        # adaptive_scaling = 0.9 * expected_std + 0.1 * local_std[:, :, ch]
        
        adapted_noise[:, :, ch] = expected_std * normalized_noise[:, :, ch]
        # adapted_noise[:, :, ch] = adaptive_scaling * normalized_noise[:, :, ch]  

    return adapted_noise


def generate_noise(grainy_image, denoised_image,shape):
    noise_map = compute_noise_map(grainy_image,denoised_image)
    # cv2.imwrite("noise_transfer/extracted_noise.png", np.clip((noise_map *25) * 255, 0, 255).astype(np.uint8))
    bin_centers, response_curve = compute_response_curve(noise_map, denoised_image)
    fitted_curves, noise_func = fit_response_curve(bin_centers, response_curve)
    P_R = np.poly1d(fitted_curves[0])  # Red channel polynomial
    P_G = np.poly1d(fitted_curves[1])  # Green channel polynomial
    P_B = np.poly1d(fitted_curves[2])
    # Step 3: Normalize Noise
    normalized_noise = normalize_with_polynomial(noise_map,[P_R,P_G,P_B])
    # cv2.imwrite("noise_transfer/normalized_noise.png", np.clip((normalized_noise *25) * 255, 0, 255).astype(np.uint8))
    denoised_patch, normalized_patch,min_coords = get_lowest_variance_patch_chunked(denoised_img,normalized_noise,patch_size=(128,128))
    
    tile_patch = tile_noise_patch(normalized_patch,shape)
    # cv2.imwrite("noise_transfer/tile_patch.png", np.clip((tile_patch *25) * 255, 0, 255).astype(np.uint8))
    cell_map, voronoi_overlay, vor,points = generate_voronoi_pattern(tile_patch,num_cells=300,border_extension=100)
    scattered_noise,translated_overlay= scatter_voronoi_cells(tile_patch, cell_map,vor,points, max_shift=20,method='voronoi_sample')
    return scattered_noise, fitted_curves, noise_func


def dasgrain_synth_pipeline(grainy_image, denoised_image, target_image,strength=10):

    th,tw,tc = target_image.shape
    scattered_noise, fitted_curves, noise_func = generate_noise(grainy_image,denoised_image,(th,tw))
    # cv2.imwrite("noise_transfer/scattered_noise.png", np.clip((scattered_noise *25) * 255, 0, 255).astype(np.uint8))
    noisy_edited = add_adaptive_noise(target_image, scattered_noise, fitted_curves, noise_func)
    
    # cv2.imwrite("noise_transfer/adapted_noise.png", np.clip((noisy_edited *25) * 255, 0, 255).astype(np.uint8))
    final_noisy_edited = np.clip((target_image + strength* noisy_edited) , 0.0, 1.0)
    # cv2.imwrite("noise_transfer/result.png", (final_noisy_edited * 255).astype(np.uint8))
    return final_noisy_edited

def process_single_image(args):
    noisy_img,denoised_img,target_path,output_path,strength = args
    target_img = cv2.imread(target_path).astype(np.float32) / 255.0
    target_noisy_img = dasgrain_synth_pipeline(noisy_img,denoised_img,target_img,strength)
    cv2.imwrite(output_path,(target_noisy_img*255).astype('uint8'))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overwrite 3-channel images with their first channel.")
    parser.add_argument("--noisy_img", type=str)
    parser.add_argument("--denoised_img", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--strength",type=float,default=10)
    args = parser.parse_args()
    noisy_img = cv2.imread(args.noisy_img).astype(np.float32) / 255.0
    denoised_img = cv2.imread(args.denoised_img).astype(np.float32) / 255.0
    
    os.makedirs(args.output_dir,exist_ok=True)
    img_pairs = []
    for i,imgs in enumerate(sorted(os.listdir(args.target_dir))):
        target_path = os.path.join(args.target_dir, imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((noisy_img,denoised_img,target_path,output_path,args.strength))
    
    num_workers = min(cpu_count(),len(img_pairs))
    # num_workers = 1
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)