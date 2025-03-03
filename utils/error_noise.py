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
from error import compute_adaptive_edge_difference

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

def crop_png(img,crop):
    x1,y1 = crop['top_left']
    x2,y2 = crop['bottom_right']
    img = img[y1:y2,x1:x2]
    return img

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
def visualize_amplified_noise_overlay(noisy, denoised,gt, amplification=5.0):
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
    gt_f = gt.astype(np.float32) / 255.0

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


def plot_relative_absolute_error(grainy, denoise,gt,crop,title,save_path):
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
    gt = gt.astype(np.float32)
    
    rae = np.abs(denoise - grainy)
    denoise_rae = np.abs(denoise-gt)
    gt_rae = np.abs(gt-grainy)
    amplification = 5.0
    gamma = 1.0
    
    amplified_noise_gt = np.clip(0.5 + amplification*gt_rae,0,1)
    amplified_noise_gt = np.power(amplified_noise_gt, gamma)

    amplified_noise_denoise = np.clip(0.5 + amplification*denoise_rae,0,1)
    amplified_noise_denoise = np.power(amplified_noise_denoise, gamma)

    
    denoise_crop = crop_png(denoise,crop)
    gt_crop = crop_png(gt,crop)
    grainy_crop = crop_png(grainy,crop)
    
    gt_rae_image = visualize_amplified_noise_overlay(grainy_crop,gt_crop,gt_crop,amplification=25)

    denoise_rae_image = visualize_amplified_noise_overlay(grainy_crop,denoise_crop,gt_crop,amplification=25)
    gt_denoise = compute_adaptive_edge_difference(gt_crop.astype('uint8'), denoise_crop.astype('uint8'))
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(gt_crop.astype('uint8'), cmap='gray')
    axes[0,1].set_title("NEAT Adjusted")
    axes[0,1].axis("off")

    axes[0,2].imshow(denoise_crop.astype('uint8'), cmap='gray')
    axes[0,2].set_title(title)
    axes[0,2].axis("off")

    axes[1,0].imshow(gt_rae_image.astype('uint8'), cmap='gray')
    axes[1,0].set_title(f"Noise/NEAT Adj")
    axes[1,0].axis("off")

    axes[1,1].imshow(denoise_rae_image.astype('uint8'), cmap='gray')
    axes[1,1].set_title(f"Noise/{title}")
    axes[1,1].axis("off")

    axes[1,2].imshow(gt_denoise.astype('float32').clip(0,1).astype('uint8'), cmap='gray')
    axes[1,2].set_title(f"Diff(NEAT Adj,{title} )")
    axes[1,2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    del fig,axes,grainy,grainy_crop,denoise,denoise_crop,gt,gt_crop, rae
    gc.collect()
    return gt_rae.mean()

def plot_ycbcr(grainy_image,denoise_image,output_path):
    ycbcr_grainy = cv2.cvtColor(grainy_image,cv2.COLOR_BGR2YCrCb)
    ycbcr_denoise = cv2.cvtColor(denoise_image,cv2.COLOR_BGR2YCrCb)

    y_g,cb_g,cr_g = cv2.split(ycbcr_grainy)
    y_d,cb_d,cr_d = cv2.split(ycbcr_denoise)
   
    scale_factor=100.0
    noise_Y = amplify_noise_residuals(y_g, y_d, scale_factor)
    noise_Cb = amplify_noise_residuals(cb_g, cb_d, scale_factor)
    noise_Cr = amplify_noise_residuals(cr_g, cr_d, scale_factor)
    y_g = enhance_grain_subtle(y_g)
    fig,axes = plt.subplots(2,2,figsize=(4,2))

    axes[0, 0].imshow(cv2.cvtColor(grainy_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Grainy Image",fontsize=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(y_g, cmap="gray")
    axes[0, 1].set_title("Y  Grainy",fontsize=8)
    axes[0, 1].axis("off")

    # Second row: Denoised Image
    axes[1, 0].imshow(cv2.cvtColor(denoise_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("NEAT Default",fontsize=8)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(noise_Y, cmap="seismic")
    axes[1, 1].set_title("Y Noise Residual",fontsize=8)
    axes[1, 1].axis("off")

    plt.tight_layout()

        # Save the figure
    plt.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(fig)
    del fig, axes, y_g, cb_g, cr_g, y_d, cb_d, cr_d, grainy_image,denoise_image
    gc.collect()

def process_single_image(args):
    grainy_path, denoise_path,gt_path,output_path,title = args
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
    gt = read_png(gt_path)
    # nre = noise_residual_energy(grainy,denoise)
    # hfen = compute_hfen(gt,denoise)
    # psnr = compute_psnr(gt,denoise)
    # psnr_grainy = compute_psnr(grainy,denoise)
    
    plot_relative_absolute_error(grainy,denoise,gt,rachel_vmd,title,output_path)
    # plot_ycbcr(grainy,denoise,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    for i,imgs in enumerate(tqdm(os.listdir(args.dir1))):
        grainy_path = os.path.join(args.dir1,imgs)
        # denoise_path = os.path.join(args.dir3,imgs)
        denoise_path = os.path.join(args.dir3,f'res_000_{i:03d}.png')
        gt_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,denoise_path,gt_path,output_path,args.title))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    
    # nre = np.mean([d["nre"] for d in results])
    # hfen = np.mean([d["hfen"] for d in results])
    # psnr = np.mean([d["psnr"] for d in results])
    # psnr_grainy = np.mean([d["psnr_grainy"] for d in results])
    # rae = np.mean([d["rae"] for d in results])
    # print(f'NRE:{nre.mean():.3f}, HFEN:{hfen.mean():.3f}, PSNR:{psnr.mean():.3f}, AE:{rae.mean():.3f}, PSNR:{psnr_grainy.mean():.3f}')