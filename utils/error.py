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

def noise_residual_energy(noisy, denoised):
    """
    Computes Noise Residual Energy (NRE).
    Lower NRE indicates better noise removal.
    """
    diff = noisy.astype(np.float32) - denoised.astype(np.float32)
    return np.linalg.norm(diff) / np.linalg.norm(noisy)

def compute_psnr(clean, denoised):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).
    Higher PSNR indicates better quality.
    """
    return metrics.peak_signal_noise_ratio(clean, denoised)

def compute_ssim(clean, denoised):
    """
    Computes Structural Similarity Index (SSIM).
    Higher SSIM indicates better perceptual similarity.
    """
    return metrics.structural_similarity(clean, denoised, data_range=denoised.max() - denoised.min(),win_size=(3,3),channel_size=-1)

def compute_hfen(clean, denoised):
    """
    Computes High-Frequency Error Norm (HFEN).
    Measures preservation of fine details using a Laplacian filter.
    Lower values indicate less loss of high-frequency components.
    """
    lap_clean = laplace(clean.astype(np.float32))
    lap_denoised = laplace(denoised.astype(np.float32))
    return np.linalg.norm(lap_clean - lap_denoised) / np.linalg.norm(lap_clean)

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


def plot_relative_absolute_error(grainy, denoise,gt,crop,title,save_path):
    """
    Computes and plots the Relative Absolute Error (RAE) between the given image and reference.

    RAE = np.abs(image - reference) / (reference + 0.0001)

    Args:
        image (numpy array): The denoised/degrained image.
        reference (numpy array): The clean ground-truth image.
    """
    # Ensure images are float32 for accurate computation
    denoise = denoise.astype(np.float32)
    grainy = grainy.astype(np.float32)
    gt = gt.astype(np.float32)
    rae = np.abs(denoise - grainy)
    gt_rae = np.abs(denoise-gt)
    gt_noise = np.abs(gt-grainy)
    amplification = 5.0
    gamma = 1.0
    amplified_noise = np.clip(0.5 + amplification*gt_noise,0,1)
    amplified_noise = np.power(amplified_noise, gamma)

    
    denoise_crop = crop_png(denoise,crop)
    gt_crop = crop_png(gt,crop)
    grainy_crop = crop_png(grainy,crop)
    rae_crop = crop_png(rae,crop)
    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt_crop.astype('uint8'), cmap='gray')
    axes[1].set_title("NEAT Adjusted")
    axes[1].axis("off")

    axes[2].imshow(denoise_crop.astype('uint8'), cmap='gray')
    axes[2].set_title(title)
    axes[2].axis("off")

    axes[3].imshow(rae_crop.clip(0,1), cmap='gray')
    # print((rae*255).astype('uint8').max())
    axes[3].set_title(f"Abs Error NEAT")
    axes[3].axis("off")
    # fig.colorbar(img_plot, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    del fig,axes,grainy,grainy_crop,denoise,denoise_crop,gt,gt_crop, rae,rae_crop
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
    nre = noise_residual_energy(grainy,denoise)
    hfen = compute_hfen(gt,denoise)
    psnr = compute_psnr(gt,denoise)
    psnr_grainy = compute_psnr(grainy,denoise)
    
    rae_val = plot_relative_absolute_error(grainy,denoise,gt,rachel_vmd,title,output_path)
    # plot_ycbcr(grainy,denoise,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
        'nre': nre,
        'hfen': hfen,
        'psnr': psnr,
        'psnr_grainy':psnr_grainy,
        'rae':rae_val
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    for imgs in tqdm(os.listdir(args.dir1)):
        grainy_path = os.path.join(args.dir1,imgs)
        denoise_path = os.path.join(args.dir3,imgs)
        gt_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,denoise_path,gt_path,output_path,args.title))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    
    nre = np.mean([d["nre"] for d in results])
    hfen = np.mean([d["hfen"] for d in results])
    psnr = np.mean([d["psnr"] for d in results])
    psnr_grainy = np.mean([d["psnr_grainy"] for d in results])
    rae = np.mean([d["rae"] for d in results])
    print(f'NRE:{nre.mean():.3f}, HFEN:{hfen.mean():.3f}, PSNR:{psnr.mean():.3f}, AE:{rae.mean():.3f}, PSNR:{psnr_grainy.mean():.3f}')