import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage
from voronoi import generate_voronoi_pattern, scatter_voronoi_cells

def compute_noise_map(noisy_image, denoised_image):
    """Extracts noise from the noisy and denoised image in the 0-1 range."""
    noise_map = noisy_image - denoised_image  # Direct subtraction, stays in 0-1 range
    return noise_map

def compute_response_curve(noise_map, denoised_image, num_bins=10):
    """
    Computes the noise response curve for each color channel in 0-1 range.

    Parameters:
    -----------
    noise_map : np.ndarray
        Noise image (noisy - denoised).
    denoised_image : np.ndarray
        Denoised base image for binning.
    num_bins : int
        Number of intensity bins to compute the response.

    Returns:
    --------
    bin_centers : np.ndarray
        Center intensity of each bin.
    response_curve : np.ndarray
        Noise std per bin for each channel.
    """
    
    h, w, c = noise_map.shape
    bin_edges = np.linspace(0, 1, num_bins + 1)  # Bins for 0-1 range
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    response_curve = np.zeros((num_bins, c))

    for i in range(num_bins):
        mask_bin = (denoised_image >= bin_edges[i]) & (denoised_image < bin_edges[i + 1])

        for ch in range(c):
            pixels_in_bin = noise_map[:, :, ch][mask_bin[:, :, ch]]
            if len(pixels_in_bin) > 1:
                response_curve[i, ch] = np.std(pixels_in_bin)
    return bin_centers, response_curve

def fit_response_curve(bin_centers, response_curve):
    """
    Fits a 3rd-degree polynomial to the response curve for each channel.

    Returns:
    --------
    fitted_curves : list
        List of polynomial coefficients per channel.
    fitted_func : function
        Polynomial function.
    """
    
    def log_func(x, a, b, c):
        x = np.clip(x, 1e-3, None)  # Ensure x is strictly positive
        return a * np.log(b * x + 1) + c  # Logarithmic fit

    def power_law_func(x, a, b, c, d):
        x = np.clip(x, 1e-3, None)  # Prevent negative or zero values
        return a * (x + b)**c + d  # Power-law fit

    def poly_func(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d  # 3rd-degree polynomial

    fitted_curves = []

    for ch in range(response_curve.shape[1]):
        y_data = response_curve[:, ch]

        valid_mask = ~np.isnan(y_data) & ~np.isinf(y_data) & (y_data > 0)
        x_valid = bin_centers[valid_mask]
        y_valid = y_data[valid_mask]

        if len(x_valid) < 4:  # If not enough valid data points, return default values
            fitted_curves.append([0, 0, 0, 0])  # 4 parameters to match power-law/poly fits
            continue

        try:
            # params, _ = curve_fit(log_func, x_valid, y_valid, p0=[5, 1, 1], 
            #                       bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=10000)
            # fitted_func = log_func
            # params, _ = curve_fit(power_law_func, x_valid, y_valid, p0=[1, 1, 1, 0],
            #                           bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]), maxfev=10000)
            # fitted_func = power_law_func
            params = np.polyfit(x_valid, y_valid, deg=3)
            fitted_func = poly_func
            fitted_curves.append(params)
        except RuntimeError:
            fitted_curves.append([0, 0, 0,0])

    return fitted_curves, fitted_func


def normalize_with_polynomial(image, polynomials, sigma=10, eps=1e-6):
    """
    Normalize an image based on fitted polynomials per channel.
    
    Args:
        image (numpy array): Input image in shape (H, W, 3) and 0-1 range.
        polynomials (list of np.poly1d): List of fitted polynomial functions [P_R, P_G, P_B].
        sigma (int): Smoothing factor for luminance estimation.
        eps (float): Small constant to prevent division by zero.

    Returns:
        numpy array: Normalized image in 0-1 range.
    """
    image = image.astype(np.float32)  # Ensure float32 format in 0-1 range

    # Compute luminance map
    luminance_map = compute_luminance(image)

    # Compute local mean luminance
    local_mean = scipy.ndimage.gaussian_filter(luminance_map, sigma=sigma)  # Shape: (H, W)

    # Normalize each channel adaptively based on the polynomial response
    normalized_image = np.zeros_like(image)
    
    for c in range(3):  # Iterate over R, G, B channels
        P_c = polynomials[c]  # Get the fitted polynomial for this channel

        # Compute the polynomial response adaptation
        response_adjusted = P_c(local_mean.flatten())  # Apply polynomial to luminance values
        response_adjusted = response_adjusted.reshape(local_mean.shape)  # Reshape to (H, W)

        # Normalize channel based on luminance and response curve
        normalized_image[..., c] = (image[..., c] - local_mean) / (response_adjusted + eps)

    # Clip and scale to ensure values remain in [0, 1] range
    normalized_image = np.clip(normalized_image * 0.5 + 0.5, 0, 1)
    return normalized_image

def normalize_noise(noise_map):
    """Normalizes noise to zero mean, unit variance per channel in the 0-1 range."""
    
    mean = np.mean(noise_map, axis=(0, 1), keepdims=True)
    std = np.std(noise_map, axis=(0, 1), keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    normalized_noise = (noise_map - mean) / std
    return normalized_noise

def compute_luminance(image):
    """Computes the luminance of an image using the Rec. 709 formula."""
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

def normalize_noise_global(noise_map):
    """Removes any pre-existing adaptiveness by normalizing noise globally to zero mean and unit variance."""
    mean = np.mean(noise_map, axis=(0, 1), keepdims=True)
    std = np.std(noise_map, axis=(0, 1), keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    return (noise_map - mean) / std

def compute_local_std(image, patch_size=16):
    """Computes local standard deviation in a sliding window manner."""
    local_std = np.zeros_like(image, dtype=np.float32)
    
    for ch in range(image.shape[2]):
        # Compute local variance using a sliding window
        mean_filter = scipy.ndimage.uniform_filter(image[:, :, ch], size=patch_size)
        mean_sq_filter = scipy.ndimage.uniform_filter(image[:, :, ch]**2, size=patch_size)
        local_std[:, :, ch] = np.sqrt(np.maximum(mean_sq_filter - mean_filter**2, 1e-6))  # Ensure numerical stability
    
    return local_std

def tile_masked_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extracts the region of `image` where `mask == 1`, tiles it to fill the
    original image shape, and returns the tiled image.

    Parameters:
    -----------
        image (H, W) or (H, W, C): Normalized input image
        mask (H, W): Binary mask (1s in the region to tile)

    Returns:
    --------
        (H, W) or (H, W, C): Tiled image using masked region
    """
    assert image.shape == mask.shape, "Image and mask must have same spatial dimensions"
    
    # Get bounding box of the mask
    ys, xs = np.where(mask[:,:,0])
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("Mask is empty or all zeros")

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # Extract patch
    patch = image[y1:y2, x1:x2]

    H, W = image.shape[:2]
    repeat_y = H // patch.shape[0] + 1
    repeat_x = W // patch.shape[1] + 1

    tiled = np.tile(patch, (repeat_y, repeat_x, 1)) if image.ndim == 3 else np.tile(patch, (repeat_y, repeat_x))
    tiled = tiled[:H, :W]  # Crop to original size

    return tiled

def add_adaptive_noise(edited_image, normalized_noise, fitted_curves, noise_func):
    """
    Adds spatially adaptive noise to each channel of an edited image based on luminance and 
    learned response curves.

    Parameters:
    -----------
    edited_image : np.ndarray
        The base image to which noise is added. Expected in [0, 1] range.
    normalized_noise : np.ndarray
        Normalized noise map to be scaled per-channel.
    fitted_curves : list
        List of fitted polynomial coefficients per channel.
    noise_func : callable
        Function that takes (luminance, *curve) and returns expected noise std.

    Returns:
    --------
    adapted_noise : np.ndarray
        The adapted noise to be added to the edited image, in [0, 1] range.
    """
    adapted_noise = np.zeros_like(edited_image, dtype=np.float32)
    luminance = compute_luminance(edited_image)  # Shared luminance for all channels

    for ch in range(edited_image.shape[2]):
        curve = fitted_curves[ch]
        expected_std = noise_func(luminance, *curve)  # Compute expected std dev
        adapted_noise[:, :, ch] = expected_std * normalized_noise[:, :, ch]

    return adapted_noise

def dasgrain_pipeline(noisy_image, denoised_image, edited_image, mask=None):
    """
    Performs DASGrain noise adaptation by extracting and adapting grain from a noisy plate
    and applying it to an edited image with optional spatial masking.

    Parameters:
    -----------
    noisy_image : np.ndarray
        Original noisy image in the range [0, 1].
    denoised_image : np.ndarray
        Denoised version of the original image.
    edited_image : np.ndarray
        Edited/composited image to which the noise will be adapted.
    mask : np.ndarray, optional
        Binary mask (same height and width) indicating edited regions. Should be in [0, 1] float format.

    Returns:
    --------
    final_noisy_edited : np.ndarray
        Edited image with adapted noise applied, in the range [0, 1].
    """

    # Step 1: Extract noise from the original image
    noise_map = compute_noise_map(noisy_image, denoised_image)

    # Step 2: Fit per-channel response curves from noise characteristics
    bin_centers, response_curve = compute_response_curve(noise_map, denoised_image)
    fitted_curves, noise_func = fit_response_curve(bin_centers, response_curve)
    polynomials = [np.poly1d(p) for p in fitted_curves]

    # Step 3: Normalize the noise using the fitted polynomial curves
    normalized_noise = normalize_with_polynomial(noise_map, polynomials)

    # Step 4: Generate Voronoi pattern based on normalized noise
    cell_map, _, vor, points = generate_voronoi_pattern(normalized_noise, num_cells=500, border_extension=100)

    # Step 5: Scatter the Voronoi cells to generate spatially adaptive noise
    scattered_noise, _ = scatter_voronoi_cells(
        normalized_noise, cell_map, vor, points,
        max_shift=20, method='voronoi_sample'
    )

    # Step 6: Apply adaptive noise to the edited image
    adapted_noise = add_adaptive_noise(edited_image, scattered_noise, fitted_curves, noise_func, mask)

    # Step 7: Combine adapted noise with original noise (outside mask) to get the final result
    if mask is None:
        final_noisy_edited = np.clip(edited_image + adapted_noise, 0.0, 1.0)
    else:
        final_noisy_edited = np.clip(edited_image + adapted_noise * mask + (1 - mask) * noise_map, 0.0, 1.0)

    return final_noisy_edited

# **Example Usage**
if __name__ == "__main__":
    noisy_image = cv2.imread("noise_transfer/noisy_vmd.png").astype(np.float32) / 255.0  # Read and normalize
    denoised_image = cv2.imread("noise_transfer/denoised_vmd.png").astype(np.float32) / 255.0
    edited_image = cv2.imread("noise_transfer/denoised_vmd.png").astype(np.float32) / 255.0
    mask = cv2.imread("noise_faulty/neat_mask1.jpg").astype(np.float32) / 255.0  # Ensure mask is also in 0-1 range

    final_result = dasgrain_pipeline(noisy_image, denoised_image, edited_image, mask)