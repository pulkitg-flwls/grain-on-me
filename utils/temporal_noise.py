import cv2
import numpy as np
import os
import torch
import multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

def load_frame_y_channel(file_path, device="cuda"):
    """Loads an image, converts to YCbCr, and extracts Y (luminance) channel."""
    image = cv2.imread(file_path)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = ycbcr[:, :, 0]  # Extract luminance channel
    # return torch.tensor(Y, dtype=torch.float32, device="cuda") if device == "cuda" and torch.cuda.is_available() else Y
    return Y

def find_textureless_patch_y_channel(Y, patch_size=32):
    """
    Finds a textureless patch in the Y (luminance) channel.

    Args:
        Y (numpy.ndarray): Grayscale Y-channel image.
        patch_size (int): Size of the square patch.

    Returns:
        tuple: (x, y) coordinates of the top-left corner of the textureless patch.
    """
    # Ensure dimensions are multiples of patch_size
    h, w = Y.shape
    h = h - (h % patch_size)
    w = w - (w % patch_size)
    Y = Y[:h, :w]  # Crop to fit patching

    # Compute local variance using a sliding window
    window_shape = (patch_size, patch_size)
    patches = view_as_windows(Y, window_shape, step=patch_size)

    # Compute variance for each patch
    variance_map = np.var(patches, axis=(2, 3))

    # Find the least textured patch (lowest variance)
    min_var_idx = np.unravel_index(np.argmin(variance_map), variance_map.shape)

    # Convert to pixel coordinates
    y, x = min_var_idx
    return x * patch_size, y * patch_size

def compute_spatial_noise(Y, patch_size=32):
    """
    Computes spatial noise standard deviation by selecting a textureless patch.

    Args:
        Y (numpy.ndarray): Grayscale Y-channel image.
        patch_size (int): Size of the square patch.

    Returns:
        float: Estimated spatial noise standard deviation.
    """
    # Find the least textured patch
    x, y = find_textureless_patch_y_channel(Y, patch_size)

    # Extract the selected patch
    patch = Y[y:y+patch_size, x:x+patch_size]

    # Compute standard deviation (proxy for noise level)
    return np.std(patch)

def compute_temporal_noise(Y_frames):
    """
    Computes temporal noise using a 5-frame rolling window (NumPy-only version).

    Args:
        Y_frames (list of numpy.ndarray): List of grayscale Y-channel frames.

    Returns:
        float: Estimated temporal noise standard deviation.
    """
    temporal_noise = []
    
    for i in range(len(Y_frames) - 4):  # 5-frame sliding window
        diffs = [np.abs(Y_frames[i + j + 1].astype(np.float32) - Y_frames[i + j].astype(np.float32)) for j in range(4)]
        avg_diff = np.std(diffs)  # Compute mean absolute difference
        temporal_noise.append(avg_diff)
    
    return np.mean(temporal_noise)  # Final noise estimate

def exponential_moving_average(spatial_noise, alpha=0.3):
    """
    Applies an exponential moving average (EMA) to smooth spatial noise.

    Args:
        spatial_noise (list of float): Spatial noise estimates per frame.
        alpha (float): Smoothing factor (0.1-0.5 works best).

    Returns:
        list: Smoothed noise estimates.
    """
    smoothed = [spatial_noise[0]]  # Initialize with first frame's noise
    for i in range(1, len(spatial_noise)):
        smoothed.append(alpha * spatial_noise[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def plot_and_save_lists(list1, list2,list3, labels=("Spatial Noise", "Smooth Noise","Gaussian Smooth"), title="Line Graph", save_path="./plot.png"):
    """
    Plots two lists as a line graph and saves the figure.

    Args:
        list1 (list of float): First data series.
        list2 (list of float): Second data series.
        labels (tuple): Labels for the two lines.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(list1, label=labels[0], marker='o', linestyle='-',color='red')
    plt.plot(list2, label=labels[1], marker='s', linestyle='-', alpha=0.7,color='green')
    plt.plot(list3, label=labels[2], marker='x', linestyle='-', alpha=0.7,color='blue')
    
    plt.xlabel("Frame Index")
    plt.ylabel("Noise Level")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(save_path, dpi=300)
    plt.close()


def profile_noise_5frame_parallel(folder_path, patch_size=32, device="cuda"):
    """Profiles noise using spatial & 5-frame temporal analysis in parallel."""
    file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])

    # Load Y channels in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        Y_frames = list(tqdm(pool.imap(load_frame_y_channel, file_list), total=len(file_list), desc="Loading frames"))


    # Compute spatial noise in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        spatial_noise = list(tqdm(pool.imap(compute_spatial_noise, Y_frames), total=len(Y_frames), desc="Computing spatial noise"))

    # Compute temporal noise using 5-frame rolling window
    # temporal_noise = compute_temporal_noise(Y_frames)

    # Final noise estimate
    # print(spatial_noise)
    spatial_std = np.mean(spatial_noise)
    total_std = exponential_moving_average(spatial_noise)
    # gaussian_std = gaussian_smooth_spatial_noise(spatial_noise)
    # print(total_std)
    # total_noise = max(spatial_std, temporal_noise)  # Conservative estimate

    # print(f"📌 Spatial Noise Std: {spatial_std:.4f}")
    # print(f"📌 Temporal Noise Std (5-frame window): {temporal_noise:.4f}")
    # print(f"🔥 Final Estimated Noise Std Dev: {total_noise:.4f}")

    # return total_noise
    # plot_and_save_lists(spatial_noise,total_std,gaussian_std)
    return total_std


def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    # os.makedirs(args.output_dir,exist_ok=True)
    noise_std = profile_noise_5frame_parallel(args.dir1)