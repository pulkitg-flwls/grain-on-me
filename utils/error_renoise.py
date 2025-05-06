import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import skimage.metrics as metrics
from skimage.filters import laplace
import gc
from multiprocessing import Pool, cpu_count
import cv2
# import pyopenexr as exr
from media_crop import detect_face_in_first_frame

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--dir3",type=str,default="")
    parser.add_argument("--title",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

# def crop_png(img,crop):
#     x1,y1 = crop['top_left']
#     x2,y2 = crop['bottom_right']
#     img = img[y1:y2,x1:x2]
#     return img
def crop_png(img,crop_coords):
    x, y, crop_width, crop_height = crop_coords
    height, width = img.shape[:2]
        
    # Ensure crop stays within image boundaries
    end_x = min(x + crop_width, width)
    end_y = min(y + crop_height, height)
    
    # Adjust crop size if it goes beyond image boundaries
    actual_width = end_x - x
    actual_height = end_y - y
    cropped_img = img[y:end_y, x:end_x]

    return cropped_img

def read_png(img_path,):
    f = iio.imread(img_path)
    # f = cv2.imread(img_path)
    return f

def amplify_noise_residuals(grainy, denoised, scale_factor=5.0):
    """
    Computes and amplifies noise residuals for better visualization.
    
    Args:
        grainy (numpy array): Grainy image channel.
        denoised (numpy array): Denoised image channel.
        scale_factor (float): Multiplier to enhance noise visibility.
        
    Returns:
        numpy array: Scaled noise residual.
    """
    noise = (grainy.astype(np.float32) - denoised.astype(np.float32)) * scale_factor
    noise = np.clip(noise, -127, 127) + 127  # Normalize to 0-255 for visualization
    return noise.astype(np.uint8)

def enhance_grain_subtle(y_channel):
    """
    Enhances grain visibility in the Y channel using adaptive contrast enhancement.
    
    Args:
        y_channel (numpy array): Y (luminance) channel from YCbCr space.
    
    Returns:
        numpy array: Enhanced grain image.
    """
    # Convert to float for precise operations
    y_channel = y_channel.astype(np.float32)
    
    # Normalize to [0,1] range
    y_norm = (y_channel - y_channel.min()) / (y_channel.max() - y_channel.min() + 1e-6)

    # Apply adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply((y_norm * 255).astype(np.uint8))

    # Convert back to float and normalize
    enhanced = y_clahe.astype(np.float32) / 255.0
    
    return (enhanced * 255).astype(np.uint8)

def high_pass_filter(image, kernel_size=3):
    """
    Applies a high-pass filter to extract fine details and grain.
    
    Args:
        image (numpy array): Input grayscale image.
        kernel_size (int): Size of the Gaussian blur kernel for high-pass filtering.
    
    Returns:
        numpy array: High-pass filtered image emphasizing noise.
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    high_pass = cv2.subtract(image, blurred) + 127  # Normalize
    return high_pass
def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
def visualize_amplified_noise_overlay(noisy, denoised,amplification=5.0):
    """
    Creates an image where the lower triangle is the denoised image with amplified noise added back,
    and the upper triangle is the clean (denoised) image.
    
    :param noisy: Noisy input image (H, W, C) in uint8
    :param denoised: Denoised output image (H, W, C) in uint8
    :param amplification: Factor to amplify the noise
    :return: Merged image with upper triangle as clean and lower triangle as amplified noisy
    """
    # Convert to float for processing
    noisy_f = noisy.astype(np.float32) / 255.0
    denoised_f = denoised.astype(np.float32) / 255.0
    # gt_f = gt.astype(np.float32) / 255.0

    # Compute noise and amplify
    noise = noisy_f - denoised_f
    # noise = normalize(noise)
    
    # amplified_noise = np.clip(gt_f + amplification * noise, 0, 1)
    amplified_noise = np.clip(amplification * noise, 0, 1)

    # Convert back to uint8
    amplified_noisy = (amplified_noise * 255).astype(np.uint8)

    # Create a mask for the lower triangle
    H, W, _ = noisy.shape
    mask = np.tri(H, W, -1, dtype=np.uint8)  # Lower triangle mask

    # Create the final blended image
    result = denoised.copy()
    result[mask == 1] = amplified_noisy[mask == 1]  # Apply amplified noisy image in lower triangle
    
    return result 


def plot_relative_absolute_error(grainy, denoise,renoise,crop,title,save_path):
    """
    Computes and plots the Relative Absolute Error (RAE) between the given image and reference.

    RAE = np.abs(image - reference) / (reference + 0.0001)

    Args:
        image (numpy array): The denoised/degrained image.
        reference (numpy array): The clean ground-truth image.
    """
    # Ensure images are float32 for accurate computation
    denoise = denoise.astype(np.float32)[:,:,:3]
    grainy = grainy.astype(np.float32)
    renoise = renoise.astype(np.float32)
    # gt = gt.astype(np.float32)
    
    # rae = np.abs(denoise - grainy)
    # denoise_rae = np.abs(denoise-gt)
    # gt_rae = np.abs(gt-grainy)
    # amplification = 5.0
    # gamma = 1.0
    
    renoise_crop = crop_png(renoise,crop)
    denoise_crop = crop_png(denoise,crop)
    grainy_crop = crop_png(grainy,crop)
    
    grainy_image = visualize_amplified_noise_overlay(grainy_crop,denoise_crop,amplification=25)

    renoise_image = visualize_amplified_noise_overlay(renoise_crop,denoise_crop,amplification=25)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    axes[0,0].set_title("Grain Slapcomp")
    # axes[0,0].set_title("Original Plate")
    axes[0,0].axis("off")

    axes[0,1].imshow(renoise_crop.astype('uint8'), cmap='gray')
    axes[0,1].set_title("Re-Noised Image")
    axes[0,1].axis("off")

    # axes[0,2].imshow(denoise_crop.astype('uint8'), cmap='gray')
    # axes[0,2].set_title(title)
    # axes[0,2].axis("off")

    axes[1,0].imshow(grainy_image.astype('uint8'), cmap='gray')
    axes[1,0].set_title(f"Noise/Grain Slapcomp")
    # axes[1,0].set_title(f"Noise/Original Plate")
    axes[1,0].axis("off")

    axes[1,1].imshow(renoise_image.astype('uint8'), cmap='gray')
    axes[1,1].set_title(f"Noise/Re-Noised {title}")
    axes[1,1].axis("off")

    # axes[1,2].imshow(gt_denoise.astype('float32').clip(0,1).astype('uint8'), cmap='gray')
    # axes[1,2].set_title(f"Diff(NEAT Adj,{title} )")
    # axes[1,2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    del fig,axes,grainy,grainy_crop,denoise,denoise_crop,renoise_crop
    gc.collect()
    
    return 


def process_single_image(args):
    grainy_path, denoise_path,renoise_path,output_path,crop_coords,title = args
    # grainy_path, denoise_path,renoise_path,output_path,title = args
    denise_ufo={
        'top_left':(1540,278),
        'bottom_right':(2052,790)
    }
    rachel_vmd = {
        'top_left':(706,5),
        'bottom_right':(1218,517)
    }
    grainy = read_png(grainy_path)
    denoise = read_png(denoise_path)
    renoise = read_png(renoise_path)
    
    
    plot_relative_absolute_error(grainy,denoise,renoise,crop_coords,title,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    crop_coords = detect_face_in_first_frame(args.dir1)
    for i,imgs in enumerate(tqdm(os.listdir(args.dir1))):
        grainy_path = os.path.join(args.dir1,imgs)
        # denoise_path = os.path.join(args.dir3,imgs)
        denoise_path = os.path.join(args.dir2,imgs)
        renoise_path = os.path.join(args.dir3,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,denoise_path,renoise_path,output_path,crop_coords,args.title))
    # for srgb vmdf02_ep01_pt03_0050
    # for i,imgs in enumerate(tqdm(range(0,120))):
    #     grainy_path = os.path.join(args.dir1,f"{imgs+1:06d}.jpg")
    #     denoise_path = os.path.join(args.dir2,f"{imgs:06d}.jpg")
    #     renoise_path = os.path.join(args.dir3,f"{imgs:06d}.jpg")
    #     # mask_path = args.mask
    #     # mask_path = os.path.join(args.mask,f"{imgs:06d}.jpg")
    #     output_path = os.path.join(args.output_dir,f"{imgs:06d}.jpg")
    #     img_pairs.append((grainy_path,denoise_path,renoise_path,output_path,crop_coords,args.title))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    
    