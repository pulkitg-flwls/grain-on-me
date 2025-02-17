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
from skimage.restoration import estimate_sigma
import gc
from multiprocessing import Pool, cpu_count
# import pyopenexr as exr

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

def read_png(img_path):
    f = iio.imread(img_path)
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


def plot_relative_absolute_error(image, reference,save_path):
    """
    Computes and plots the Relative Absolute Error (RAE) between the given image and reference.

    RAE = np.abs(image - reference) / (reference + 0.0001)

    Args:
        image (numpy array): The denoised/degrained image.
        reference (numpy array): The clean ground-truth image.
    """
    # Ensure images are float32 for accurate computation
    image = image.astype(np.float32)
    reference = reference.astype(np.float32)
    
    # Compute Relative Absolute Error (RAE)
    im1 = image/255.0
    ref1 = reference/255.0
    
    rae = np.abs(image - reference)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(reference.astype('uint8'), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image.astype('uint8'), cmap='gray')
    axes[1].set_title("NEAT Default")
    axes[1].axis("off")

    
    axes[2].imshow(rae.clip(0,1), cmap='jet')
    # print((rae*255).astype('uint8').max())
    axes[2].set_title(f"AE:{rae.mean():.3f}")
    axes[2].axis("off")
    # fig.colorbar(img_plot, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    del fig,axes,image,reference
    gc.collect()
    return rae.mean()

def process_single_image(args):
    grainy_path, denoise_path,output_path = args
    grainy = read_png(grainy_path)
    denoise = read_png(denoise_path)
    nre = noise_residual_energy(grainy,denoise)
    hfen = compute_hfen(grainy,denoise)
    psnr = compute_psnr(grainy,denoise)
    rae_val = plot_relative_absolute_error(grainy,denoise,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
        'nre': nre,
        'hfen': hfen,
        'psnr': psnr,
        'rae':rae_val
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    for imgs in tqdm(os.listdir(args.dir1)[:10]):
        grainy_path = os.path.join(args.dir1,imgs)
        denoise_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,denoise_path,output_path))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    
    nre = np.mean([d["nre"] for d in results])
    hfen = np.mean([d["hfen"] for d in results])
    psnr = np.mean([d["psnr"] for d in results])
    rae = np.mean([d["rae"] for d in results])
    # nre = np.asarray(nre)
    # hfen = np.asarray(hfen)
    # psnr = np.asarray(psnr)
    # rae = np.asarray(rae)
    print(f'NRE:{nre.mean():.3f}, HFEN:{hfen.mean():.3f}, PSNR:{psnr.mean():.3f}, AE:{rae.mean():.3f}')