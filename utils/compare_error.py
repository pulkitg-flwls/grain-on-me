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
from error_noise import visualize_amplified_noise_overlay

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

def plot_relative_absolute_error(grainy,gt,neat,bil,bm3d,nlm,crop,save_path):
    """
    Computes and plots the Relative Absolute Error (RAE) between the given image and reference.

    RAE = np.abs(image - reference) / (reference + 0.0001)

    Args:
        image (numpy array): The denoised/degrained image.
        reference (numpy array): The clean ground-truth image.
    """
    # Ensure images are float32 for accurate computation
    grainy = grainy.astype(np.float32)
    gt = gt.astype(np.float32)
    neat = neat.astype(np.float32)
    bil = bil.astype(np.float32)
    bm3d = bm3d.astype(np.float32)
    nlm = nlm.astype(np.float32)
        
    grainy_crop = crop_png(grainy,crop)
    gt_crop = crop_png(gt,crop)
    neat_crop = crop_png(neat,crop)
    bil_crop = crop_png(bil,crop)
    nlm_crop = crop_png(nlm,crop)
    bm3d_crop = crop_png(bm3d,crop)
    
    gt_rae_image = visualize_amplified_noise_overlay(grainy_crop,gt_crop,gt_crop,amplification=25)
    neat_rae_image = visualize_amplified_noise_overlay(grainy_crop,neat_crop,gt_crop,amplification=25)
    bil_rae_image = visualize_amplified_noise_overlay(grainy_crop,bil_crop,gt_crop,amplification=25)
    nlm_rae_image = visualize_amplified_noise_overlay(grainy_crop,nlm_crop,gt_crop,amplification=25)
    bm3d_rae_image = visualize_amplified_noise_overlay(grainy_crop,bm3d_crop,gt_crop,amplification=25)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(neat_rae_image.astype('uint8'), cmap='gray')
    axes[0,1].set_title("Noise/NEAT Default")
    axes[0,1].axis("off")

    axes[0,2].imshow(bil_rae_image.astype('uint8'), cmap='gray')
    axes[0,2].set_title("Noise/Bilateral")
    axes[0,2].axis("off")

    axes[1,0].imshow(gt_rae_image.astype('uint8'), cmap='gray')
    axes[1,0].set_title(f"Noise/NEAT Adj")
    axes[1,0].axis("off")

    axes[1,1].imshow(nlm_rae_image.astype('uint8'), cmap='gray')
    axes[1,1].set_title(f"Noise/NLM Temporal")
    axes[1,1].axis("off")

    axes[1,2].imshow(bm3d_rae_image.astype('uint8'), cmap='gray')
    axes[1,2].set_title(f"Noise/BM3D Temporal")
    axes[1,2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    del fig,axes,grainy,grainy_crop,gt,gt_crop,nlm,nlm_crop,bm3d,bm3d_crop,neat,neat_crop,bil,bil_crop
    gc.collect()

def process_single_image(args):
    grainy_path,gt_path,neat_path,bil_path,bm3d_path,nlm_path, save_path = args
    denise_ufo={
        'top_left':(1540,278),
        'bottom_right':(2052,790)
    }
    rachel_vmd = {
        'top_left':(706,5),
        'bottom_right':(1218,517)
    }
    grainy = read_png(grainy_path)
    gt = read_png(gt_path)
    neat = read_png(neat_path)
    bil = read_png(bil_path)
    bm3d = read_png(bm3d_path)
    nlm = read_png(nlm_path)

    plot_relative_absolute_error(grainy,gt,neat,bil,bm3d,nlm,rachel_vmd,save_path)
    # plot_ycbcr(grainy,denoise,output_path)
    return {
        'img_name': grainy_path.split('/')[-1],
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    neat_dir = "../data/resize_4x/vmdf02/ep01_pt37_0140/DENOISED/png/png/"
    bil_dir = "../data/results/bilateral/vmdf02/ep01_pt37_0140/"
    bm3d_dir = "../data/results/bm3d_temp/vmdf02/ep01_pt37_0140/"
    nlm_dir = "../data/results/nlm_std_temp/vmdf02/ep01_pt37_0140/"

    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    for imgs in tqdm(os.listdir(args.dir1)):
        grainy_path = os.path.join(args.dir1,imgs)
        gt_path = os.path.join(args.dir2,imgs)
        neat_path = os.path.join(neat_dir,imgs)
        bil_path = os.path.join(bil_dir,imgs)
        bm3d_path = os.path.join(bm3d_dir,imgs)
        nlm_path = os.path.join(nlm_dir,imgs)
        
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,gt_path,neat_path,bil_path,bm3d_path,nlm_path,output_path))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    