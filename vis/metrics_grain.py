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
from skimage.metrics import structural_similarity as ssim
import flip_evaluator as flip
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
    f = iio.imread(img_path)/255.0
    # f = cv2.imread(img_path)
    return f[:,:,:3]

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
    ssim= metrics.structural_similarity(clean, denoised,data_range = 1,channel_axis=-1)
    return ssim

def compute_mae(clean, denoised):
    """
    Computes MAE.
    Lower MAE.
    """
    mae = np.mean(np.abs(clean-denoised))
    return mae

def compute_flip(clean,denoised):
    flipErrorMap, meanFLIPError, parameters = flip.evaluate(clean, denoised,'LDR')
    return meanFLIPError
import signal

def handler(signum, frame):
    raise TimeoutError("Processing took too long!")
def process_single_image(args):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(150)
    try:
        grainy_path, denoise_path, = args
        
        grainy = read_png(grainy_path)
        regrain = read_png(denoise_path)
        psnr = compute_psnr(grainy,regrain)
        ssim = compute_ssim(grainy,regrain)
        mae = compute_mae(grainy,regrain)
        flipy = compute_flip(grainy,regrain)
        return {
            'img_name': grainy_path.split('/')[-1],
            'ssim': ssim,
            'psnr': psnr,
            'mae': mae,
            'flip': flipy
        }
    except TimeoutError:
        print(f"Timeout: {grainy_path}")
        return None

def image_pairs_generator(args):
    for imgs in sorted(os.listdir(args.dir1))[:90]:
        grainy_path = os.path.join(args.dir1,imgs)
        regrain_path = os.path.join(args.dir2,imgs)
        yield(grainy_path,regrain_path)


if __name__=="__main__":
    args = parse_args()
    # os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    num_workers = min(cpu_count(), 8)
    print("Num Workers:", num_workers)
    batch_size = 50
    results = []
    with Pool(processes=num_workers) as pool:
        gen = image_pairs_generator(args)
        while True:
            batch = list(next(gen, None) for _ in range(batch_size))
            batch = [b for b in batch if b is not None]  # Remove None values

            if not batch:
                break  # Stop when generator is exhausted

            for res in tqdm(pool.imap_unordered(process_single_image, batch), total=len(batch), desc="Processing batch"):
                if res is not None:
                    results.append(res)  # Collect results
    
    # nre = np.mean([d["nre"] for d in results])
    # hfen = np.mean([d["hfen"] for d in results])
    psnr = np.mean([d["psnr"] for d in results])
    ssimy = np.mean([d["ssim"] for d in results])
    mae = np.mean([d["mae"] for d in results])
    flipy = np.mean([d["flip"] for d in results])
   
    print(f'PSNR:{psnr.mean():.3f}, SSIM:{ssimy.mean():.3f}, MAE:{mae.mean():.5f},FLIP:{flipy.mean():.4f}')