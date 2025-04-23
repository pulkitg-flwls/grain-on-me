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
    ssim= metrics.structural_similarity(clean, denoised,channel_axis=-1)
    return ssim

def compute_hfen(clean, denoised):
    """
    Computes High-Frequency Error Norm (HFEN).
    Measures preservation of fine details using a Laplacian filter.
    Lower values indicate less loss of high-frequency components.
    """
    lap_clean = laplace(clean.astype(np.float32))
    lap_denoised = laplace(denoised.astype(np.float32))
    return np.linalg.norm(lap_clean - lap_denoised) / np.linalg.norm(lap_clean)

def compute_mae(clean, denoised):
    """
    Computes MAE.
    Lower MAE.
    """
    mae = np.abs(clean,denoised).mean()
    return mae

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

def visualize_amplified_noise_overlay(noisy, denoised, amplification=5.0):
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

    # Compute noise and amplify
    noise = noisy_f - denoised_f
    amplified_noise = np.clip(denoised_f + amplification * noise, 0, 1)

    # Convert back to uint8
    amplified_noisy = (amplified_noise * 255).astype(np.uint8)

    # Create a mask for the lower triangle
    H, W, _ = noisy.shape
    mask = np.tri(H, W, -1, dtype=np.uint8)  # Lower triangle mask

    # Create the final blended image
    result = denoised.copy()
    result[mask == 1] = amplified_noisy[mask == 1]  # Apply amplified noisy image in lower triangle
    
    return result 


def adaptive_contrast_enhancement(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
    
    :param image: Grayscale image
    :return: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def detect_face_region(image):
    """
    Detects the face region and returns a binary mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    mask = np.zeros_like(gray, dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 1
    return mask

def compute_adaptive_edge_difference(gt, pred, amplification=5.0):
    """
    Computes an enhanced, adaptive edge difference map with contrast boosting.
    """
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    # Enhance contrast in the ground truth to better detect lost details
    gt_gray = adaptive_contrast_enhancement(gt_gray)
    pred_gray = adaptive_contrast_enhancement(pred_gray)

    # Apply Canny edge detection (extracts high-frequency details)
    edges_gt = cv2.Canny(gt_gray, 50, 100)
    edges_pred = cv2.Canny(pred_gray, 50, 100)
    median_intensity = np.median(gt_gray)
    # canny_low = int(max(10, 0.33 * median_intensity))  # Adaptive lower bound
    # canny_high = int(min(255, 1.33 * median_intensity))  # Adaptive upper bound
    # edges_gt = cv2.Canny(gt_gray, canny_low, canny_high)
    # edges_pred = cv2.Canny(pred_gray, canny_low, canny_high)
    # Compute absolute edge difference
    edge_difference = np.abs(edges_gt - edges_pred)

    # Gaussian-based contrast normalization
    gaussian_weight = cv2.GaussianBlur(edge_difference.astype(np.float32), (7,7), 2.0)
    edge_difference = (edge_difference * gaussian_weight) ** 0.5  # Adaptive contrast boost

    # Apply face mask to suppress background noise
    face_mask = detect_face_region(gt)
    # edge_difference = edge_difference * face_mask

    # Amplify and normalize
    edge_difference *= amplification
    edge_difference = np.clip(edge_difference, 1e-5, 255)  # Avoid log(0)
    edge_difference = np.log1p(edge_difference)
    edge_difference = (edge_difference - edge_difference.min()) / (edge_difference.max() - edge_difference.min())

    return edge_difference

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

    
    # denoise_crop = crop_png(denoise,crop)
    # gt_crop = crop_png(gt,crop)
    # grainy_crop = crop_png(grainy,crop)
    # rae_crop = crop_png(rae,crop)
    # rae_crop = crop_png(amplified_noise,crop)
    # rae_image = visualize_amplified_noise_overlay(grainy_crop,gt_crop)
    # rae_image = compute_adaptive_edge_difference(gt_crop.astype('uint8'), denoise_crop.astype('uint8'))
    # # Plot results
    
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # axes[0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")

    # axes[1].imshow(gt_crop.astype('uint8'), cmap='gray')
    # axes[1].set_title("NEAT Adjusted")
    # axes[1].axis("off")

    # axes[2].imshow(denoise_crop.astype('uint8'), cmap='gray')
    # axes[2].set_title(title)
    # axes[2].axis("off")

    # axes[3].imshow(rae_image.astype('float32').clip(0,1), cmap='gray')
    # # print((rae*255).astype('uint8').max())
    # axes[3].set_title(f"Abs Error NEAT")
    # axes[3].axis("off")
    # # fig.colorbar(img_plot, ax=axes[2], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.savefig(save_path, dpi=400, bbox_inches='tight')
    # del fig,axes,grainy,grainy_crop,denoise,denoise_crop,gt,gt_crop, rae
    # gc.collect()
    return gt_rae.mean()


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
    ssim = compute_ssim(gt,denoise)
    psnr_grainy = compute_psnr(grainy,denoise)
    mae = compute_mae(grainy,denoise)
    # rae_val = plot_relative_absolute_error(grainy,denoise,gt,rachel_vmd,title,output_path)
    # plot_ycbcr(grainy,denoise,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
        'nre': nre,
        'hfen': hfen,
        'psnr': psnr,
        'ssim': ssim,
        'psnr_grainy':psnr_grainy,
        'mae':mae
    }
    

if __name__=="__main__":
    args = parse_args()
    # os.makedirs(args.output_dir,exist_ok=True)
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
    ssim = np.mean([d["ssim"] for d in results])
    psnr_grainy = np.mean([d["psnr_grainy"] for d in results])
    mae = np.mean([d["mae"] for d in results])
    print(f'NRE:{nre.mean():.3f}, HFEN:{hfen.mean():.3f}, PSNR:{psnr.mean():.3f}, \
          MAE:{mae.mean():.3f}, PSNR_Grainy:{psnr_grainy.mean():.3f}, SSIM: {ssim.mean():.3f}')