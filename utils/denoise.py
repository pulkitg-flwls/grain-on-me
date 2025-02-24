import os
import imageio.v3 as iio
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count
import cv2
from skimage.util import view_as_windows
from spatial import wavelet_denoise_y_channel,bilateral_filter_luminance,\
    bm3d_denoise_y_channel_gpu, nlm_denoise_y_channel
from temporal_noise import profile_noise_5frame_parallel

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

def read_png(img_path):
    # f = iio.imread(img_path)
    f = cv2.imread(img_path)
    return f



def find_textureless_patch_y_channel(image, patch_size=32):
    """
    Finds a textureless patch in the Y (luminance) channel of an image.

    Args:
        image (numpy.ndarray): Input RGB image.
        patch_size (int): Size of the square patch.

    Returns:
        tuple: (x, y) coordinates of the top-left corner of the textureless patch.
    """
    # Convert to YCbCr color space
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Extract only the Y (luminance) channel
    Y = ycbcr[:, :, 0]

    # Compute local variance using a sliding window
    window_shape = (patch_size, patch_size)
    patches = view_as_windows(Y, window_shape, step=patch_size)

    # Compute variance for each patch
    variance_map = np.var(patches, axis=(2, 3))  # Variance of each patch

    # Find the patch with the lowest variance
    min_var_idx = np.unravel_index(np.argmin(variance_map), variance_map.shape)

    # Convert index to pixel coordinates
    y, x = min_var_idx
    return x * patch_size, y * patch_size

def profile_noise_y_channel(image, x, y, patch_size=32):
    """
    Estimates noise characteristics from a textureless patch in the Y channel.

    Args:
        image (numpy.ndarray): Input RGB image.
        x (int): X coordinate of the patch.
        y (int): Y coordinate of the patch.
        patch_size (int): Size of the square patch.

    Returns:
        dict: Noise statistics including mean, standard deviation, and frequency spectrum.
    """
    # Convert to YCbCr and extract Y channel
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = ycbcr[:, :, 0]

    # Extract the selected patch
    patch = Y[y:y+patch_size, x:x+patch_size]

    # Compute noise mean and standard deviation
    mean_noise = np.mean(patch)
    std_noise = np.std(patch)

    # Compute frequency-domain representation (Fourier Transform)
    dft = np.fft.fft2(patch)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    return {
        "mean": mean_noise,
        "std_dev": std_noise,
        "frequency_spectrum": magnitude_spectrum
    }

def process_single_image(args):
    
    grainy_path, denoise_path,output_path,noise_profile = args
    grainy = read_png(grainy_path)
    denoise = read_png(denoise_path)
    # nre = noise_residual_energy(grainy,denoise)
    # hfen = compute_hfen(grainy,denoise)
    # psnr = compute_psnr(grainy,denoise)
    # rae_val = plot_relative_absolute_error(grainy,denoise,output_path)
    # bilateral_filter_luminance(grainy,output_path)
    # wavelet_denoise_y_channel(grainy,output_path)]
    # x_patch, y_patch = find_textureless_patch_y_channel(grainy,patch_size=32)
    # noise = profile_noise_y_channel(grainy,x_patch,y_patch,patch_size=32)
    bm3d_denoise_y_channel_gpu(grainy,output_path,sigma=noise_profile)
    # wavelet_denoise_y_channel(grainy,output_path,std_dev=noise['std_dev'])
    # nlm_denoise_y_channel(grainy,output_path,std_dev=noise_profile)
    return {
        'img_name': grainy_path.split('/')[-1],
        # 'nre': nre,
        # 'hfen': hfen,
        # 'psnr': psnr,
        # 'rae':rae_val
    }
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    noise_profile = profile_noise_5frame_parallel(args.dir1)
    for i,imgs in tqdm(enumerate(os.listdir(args.dir1))):
        grainy_path = os.path.join(args.dir1,imgs)
        denoise_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((grainy_path,denoise_path,output_path,noise_profile[i]))
        # process_single_image(img_pairs)
        # break
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    
    # nre = np.mean([d["nre"] for d in results])
    # hfen = np.mean([d["hfen"] for d in results])
    # psnr = np.mean([d["psnr"] for d in results])
    # rae = np.mean([d["rae"] for d in results])
    # print(f'NRE:{nre.mean():.3f}, HFEN:{hfen.mean():.3f}, PSNR:{psnr.mean():.3f}, AE:{rae.mean():.3f}')