import cv2
import pywt
import torch
import bm3d
import torch.nn.functional as F
import cv2
import numpy as np

def bilateral_filter_luminance(image,output_path, d=7, sigma_spatial=15, sigma_range=30):
    """
    Applies bilateral filtering on the Y channel of an image in YCbCr space.

    Args:
        image (numpy.ndarray): Input RGB image.
        d (int): Diameter of the pixel neighborhood.
        sigma_spatial (float): Spatial Gaussian standard deviation (controls neighborhood size).
        sigma_range (float): Intensity Gaussian standard deviation (controls edge preservation).

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y,cb,cr = cv2.split(ycbcr)
    y_filtered = cv2.bilateralFilter(y, d, float(sigma_range), float(sigma_spatial))
    ycbcr_filtered = cv2.merge([y_filtered,cb,cr])
    denoised_rgb = cv2.cvtColor(ycbcr_filtered,cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path,denoised_rgb)

def wavelet_denoise_y_channel(image,output_path, wavelet='db2',std_dev=10, level=3):
    """
    Applies wavelet-based denoising on the Y (luminance) channel in YCbCr space.

    Args:
        image (numpy.ndarray): Input RGB image.
        wavelet (str): Type of wavelet ('haar', 'db1', 'sym4', etc.).
        level (int): Number of wavelet decomposition levels.

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Convert RGB to YCbCr
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split Y, Cb, Cr channels
    Y, Cb, Cr = cv2.split(ycbcr)

    # Normalize Y to [0,1] for better processing
    Y = np.float32(Y) / 255.0

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(Y, wavelet, level=level)
    coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    # Estimate noise level (using Median Absolute Deviation)
    # sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    sigma = std_dev*2
    # Calculate soft threshold
    threshold = sigma * np.sqrt(2 * np.log(Y.size))

    # Apply soft thresholding to high-frequency wavelet coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
    for detail_coeffs in coeffs[1:]:  # Process (LH, HL, HH) tuples
        coeffs_thresh.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in detail_coeffs))

    # Reconstruct the denoised Y channel
    Y_denoised = pywt.waverec2(coeffs_thresh, wavelet)

    # Clip and rescale to 8-bit
    Y_denoised = np.clip(Y_denoised * 255, 0, 255).astype(np.uint8)

    # Merge denoised Y with original Cb, Cr
    ycbcr_denoised = cv2.merge([Y_denoised, Cb, Cr])

    # Convert back to RGB
    denoised_rgb = cv2.cvtColor(ycbcr_denoised, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path,denoised_rgb)
    del Y_denoised

def bm3d_denoise_y_channel_gpu(image,output_path, sigma=25, device="cuda"):
    """
    Applies BM3D denoising on the Y channel of an image using GPU acceleration.

    Args:
        image (numpy.ndarray): Input RGB image.
        sigma (float): Noise standard deviation.
        device (str): 'cuda' for GPU acceleration, 'cpu' otherwise.

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    if not torch.cuda.is_available():
        device = "cpu"
        print("⚠️ CUDA not available! Falling back to CPU.")

    # Convert to YCbCr
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split channels
    Y, Cb, Cr = cv2.split(ycbcr)

    # Normalize Y to [0,1] and move to GPU
    Y = torch.tensor(Y, dtype=torch.float32, device=device) / 255.0

    # Apply BM3D on GPU
    with torch.no_grad():
        Y_denoised = bm3d.bm3d(Y.cpu().numpy(), sigma_psd=sigma / 255.0, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
    # Move back to GPU for post-processing
    Y_denoised = torch.tensor(Y_denoised, dtype=torch.float32, device=device)

    # Convert back to 8-bit
    Y_denoised = torch.clamp(Y_denoised * 255, 0, 255).byte().cpu().numpy()

    # Merge denoised Y with original Cb, Cr
    ycbcr_denoised = cv2.merge([Y_denoised, Cb, Cr])

    # Convert back to RGB
    denoised_rgb = cv2.cvtColor(ycbcr_denoised, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path,denoised_rgb)
    # return denoised_rgb

def nlm_denoise_torch(image,output_path, std_dev, patch_size=3, search_window=21, device="cuda"):
    """
    Applies CUDA-accelerated Non-Local Means (NLM) denoising using PyTorch.

    Args:
        image (numpy.ndarray): Input RGB image.
        std_dev (float): Estimated noise standard deviation.
        patch_size (int): Size of the local patch.
        search_window (int): Size of the search window.
        device (str): "cuda" for GPU acceleration, "cpu" otherwise.

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available! Falling back to CPU.")
        device = "cpu"

    # Convert RGB to YCbCr and extract Y channel
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(ycbcr)

    # Convert to Torch tensor and move to GPU
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device) / 255.0
    Y_tensor = Y_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1,1,H,W)

    # Extract patches using unfold (efficient memory access)
    padding = patch_size // 2
    Y_patches = F.unfold(Y_tensor, kernel_size=patch_size, padding=padding)  # Shape: (1, patch_size^2, num_patches)

    # Compute similarity weights
    distances = torch.cdist(Y_patches.transpose(1, 2), Y_patches.transpose(1, 2), p=2)  # Euclidean distance
    weights = torch.exp(-distances / (std_dev ** 2))  # Gaussian weight function
    weights /= weights.sum(dim=-1, keepdim=True)  # Normalize weights

    # Apply weighted averaging
    Y_denoised = (weights @ Y_patches.transpose(1, 2)).mean(dim=-1)

    # Reshape back to image dimensions
    Y_denoised = Y_denoised.view(Y_tensor.shape[-2:])

    # Convert back to 8-bit
    Y_denoised = torch.clamp(Y_denoised * 255, 0, 255).byte().cpu().numpy()

    # Merge denoised Y with original Cb, Cr
    ycbcr_denoised = cv2.merge([Y_denoised, Cb, Cr])

    denoised_rgb = cv2.cvtColor(ycbcr_denoised, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path,denoised_rgb)
    del denoised_rgb

def nlm_denoise_y_channel(image,output_path, std_dev):
    """
    Applies Non-Local Means (NLM) denoising on the Y (luminance) channel.

    Args:
        image (numpy.ndarray): Input RGB image.
        std_dev (float): Estimated noise standard deviation.

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    # Convert to YCbCr color space
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split channels
    Y, Cb, Cr = cv2.split(ycbcr)

    # Apply Non-Local Means Denoising (Adaptive based on std_dev)
    h = std_dev * 2  # Strength of filtering
    Y_denoised = cv2.fastNlMeansDenoising(Y, None, h, 7, 21)

    # Merge back with original Cb, Cr channels
    ycbcr_denoised = cv2.merge([Y_denoised, Cb, Cr])

    # Convert back to RGB
    denoised_rgb = cv2.cvtColor(ycbcr_denoised, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path,denoised_rgb)