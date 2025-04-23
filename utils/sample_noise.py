import cv2
import numpy as np
from scipy import ndimage
from skimage.util import view_as_windows

def sample_textureless_noise(noise, denoised_img, window_size=16, top_n_patches=10):
    """
    Find the most textureless patches and sample noise from them
    
    Parameters:
    -----------
    noisy_img : numpy.ndarray
        Original noisy image
    denoised_img : numpy.ndarray
        Denoised version of the image
    window_size : int
        Size of patches to analyze (default: 16)
    top_n_patches : int
        Number of most textureless patches to use
        
    Returns:
    --------
    tuple
        (noise_samples, texture_scores) - Noise samples from textureless areas and their texture scores
    """
    # Convert to grayscale for texture analysis
    if len(denoised_img.shape) == 3:
        gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = denoised_img.copy()
    
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    
    if np.max(gray) > 1.0:
        gray = gray / 255.0
    
    # Extract noise
    # noise = noisy_img.astype(np.float32) - denoised_img.astype(np.float32)
    
    # Calculate gradient magnitude for texture detection (optimized with Sobel)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_mag = np.hypot(gx, gy)
    
    # Calculate local standard deviation for texture detection
    std_dev = ndimage.generic_filter(gray, np.std, size=3)
    
    # Combine metrics (lower = less texture)
    texture_metric = grad_mag + 5 * std_dev
    
    # Extract windows efficiently with view_as_windows
    # Ensure dimensions are compatible with window_size
    h, w = texture_metric.shape
    h_valid = h - (h % window_size)
    w_valid = w - (w % window_size)
    
    if h_valid < window_size or w_valid < window_size:
        # Image too small, adjust window size
        window_size = min(h_valid, w_valid)
        if window_size < 4:  # Too small for meaningful analysis
            window_size = 4
            h_valid = h - (h % window_size)
            w_valid = w - (w % window_size)
    
    # Crop to valid dimensions
    texture_metric = texture_metric[:h_valid, :w_valid]
    noise_cropped = noise[:h_valid, :w_valid]
    
    # Extract patches using view_as_windows (highly optimized)
    texture_patches = view_as_windows(texture_metric, (window_size, window_size), step=window_size)
    texture_scores = np.mean(texture_patches, axis=(2, 3)).flatten()
    
    # Find indices of the most textureless patches
    top_indices = np.argsort(texture_scores)[:top_n_patches]
    
    # Calculate patch positions in original image
    patch_positions = []
    patches_y, patches_x = texture_patches.shape[:2]
    for idx in top_indices:
        y_idx = idx // patches_x
        x_idx = idx % patches_x
        patch_positions.append((y_idx * window_size, x_idx * window_size))
    
    # Extract noise from these patches
    noise_samples = []
    for y, x in patch_positions:
        patch_noise = noise_cropped[y:y+window_size, x:x+window_size]
        noise_samples.append(patch_noise)
    
    return noise_samples, texture_scores[top_indices]

def compute_noise_statistics(noise_samples):
    """
    Compute statistics from noise samples
    
    Parameters:
    -----------
    noise_samples : list
        List of noise patches
    
    Returns:
    --------
    dict
        Dictionary of noise statistics
    """
    # Stack all samples
    all_samples = np.vstack([sample.reshape(-1, sample.shape[-1]) for sample in noise_samples])
    
    # Compute basic statistics
    stats = {
        'mean': np.mean(all_samples, axis=0),
        'std': np.std(all_samples, axis=0),
        'min': np.min(all_samples, axis=0),
        'max': np.max(all_samples, axis=0)
    }
    
    # Compute histogram for each channel
    hist_range = (-0.5, 0.5)  # Typical range for noise
    bins = 100
    histograms = []
    
    for c in range(all_samples.shape[1]):
        hist, bin_edges = np.histogram(all_samples[:, c], bins=bins, range=hist_range, density=True)
        histograms.append((hist, bin_edges))
    
    stats['histograms'] = histograms
    
    # Compute power spectral density if samples are large enough
    if len(noise_samples) > 0 and noise_samples[0].shape[0] >= 8:
        # Take a representative patch for spectral analysis
        sample_patch = noise_samples[0]
        channels = sample_patch.shape[2] if len(sample_patch.shape) == 3 else 1
        
        psd_list = []
        for c in range(channels):
            if channels == 1:
                patch_c = sample_patch
            else:
                patch_c = sample_patch[:, :, c]
            
            # Compute FFT and power spectrum
            fft = np.fft.fft2(patch_c)
            fft_shifted = np.fft.fftshift(fft)
            psd = np.abs(fft_shifted) ** 2
            psd_norm = psd / np.sum(psd)
            psd_list.append(psd_norm)
        
        stats['psd'] = psd_list
    
    return stats

def apply_sampled_noise(image, mask, noise_stats, seed=None):
    """
    Apply noise with the given statistics to the masked regions
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image to apply noise to
    mask : numpy.ndarray
        Mask where >0 indicates regions to apply noise
    noise_stats : dict
        Statistics of the noise to apply
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    numpy.ndarray
        Image with noise applied to masked regions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Prepare mask
    if len(mask.shape) == 2 and len(image.shape) == 3:
        mask_3ch = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
    else:
        mask_3ch = mask.copy()
    
    mask_float = mask_3ch.astype(np.float32) / np.max(mask_3ch)
    
    # Generate noise with matching statistics
    shape = image.shape
    synthetic_noise = np.zeros(shape, dtype=np.float32)
    
    for c in range(shape[2]):
        # Generate Gaussian noise with matching stats
        noise_c = np.random.normal(noise_stats['mean'][c], noise_stats['std'][c], size=(shape[0], shape[1]))
        
        # Optional: Match histogram exactly using inverse transform sampling
        if 'histograms' in noise_stats:
            hist, bin_edges = noise_stats['histograms'][c]
            # Convert to CDF
            cdf = np.cumsum(hist) / np.sum(hist)
            # Generate uniform random values
            uniform_samples = np.random.uniform(0, 1, size=(shape[0], shape[1]))
            # Apply inverse transform
            bin_indices = np.searchsorted(cdf, uniform_samples.flatten())
            # Map to bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            noise_c = bin_centers[bin_indices].reshape(shape[0], shape[1])
        
        synthetic_noise[:, :, c] = noise_c
    
    # Apply noise to masked regions (additive blending)
    result = image.copy()
    result = result + synthetic_noise * mask_float
    
    # Clip to valid range
    return np.clip(result, 0, 1)

# Example usage
def add_scattered_grain(noise, denoised_img, edited_img, mask, window_size=16):
    """
    Add scattered grain to edited regions based on textureless noise from original
    
    Parameters:
    -----------
    noisy_img : numpy.ndarray
        Original noisy image
    denoised_img : numpy.ndarray
        Denoised version of original image
    edited_img : numpy.ndarray
        Edited image to apply grain to
    mask : numpy.ndarray
        Mask of edited regions
    window_size : int
        Size of patches for analysis
    
    Returns:
    --------
    numpy.ndarray
        Result with matched grain
    """
    # Find textureless patches and extract noise statistics
    noise_samples, texture_scores = sample_textureless_noise(
        noise, denoised_img, window_size=window_size
    )
    
    # Compute noise statistics
    noise_stats = compute_noise_statistics(noise_samples)
    
    # Apply noise to edited regions
    result = apply_sampled_noise(edited_img, mask, noise_stats)
    
    return result, noise_stats