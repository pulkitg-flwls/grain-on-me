import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage
from voronoi import generate_voronoi_pattern, scatter_voronoi_cells

def compute_noise_map(noisy_image, denoised_image):
    """Extracts noise from the noisy and denoised image in the 0-1 range."""
    noise_map = noisy_image - denoised_image  # Direct subtraction, stays in 0-1 range
    cv2.imwrite("noise_faulty/extracted_noise.png", np.clip((noise_map *25) * 255, 0, 255).astype(np.uint8))  # Save for visualization
    return noise_map

def compute_response_curve(noise_map, denoised_image, num_bins=10):
    """Computes the noise response curve for each color channel in 0-1 range."""
    
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
    # print(bin_centers,response_curve)
    return bin_centers, response_curve

def fit_response_curve(bin_centers, response_curve):
    """Fits a smooth function to the noise response curve."""
    
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

        # if len(x_valid) < 3:  
        #     fitted_curves.append([0, 0, 0])
        #     continue
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

def compute_luminance(image):
    """
    Compute luminance from the LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[..., 0]  # Extract the L (lightness) channel
    return luminance

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
        image (H, W) or (H, W, C): Normalized input image
        mask (H, W): Binary mask (1s in the region to tile)

    Returns:
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

def add_adaptive_noise(edited_image, normalized_noise, fitted_curves, noise_func, mask=None):
    """Adds adapted noise to the edited image using the response curve in 0-1 range."""
    
    adapted_noise = np.zeros_like(edited_image, dtype=np.float32)
    luminance = compute_luminance(edited_image)  # Extract the luminance
    local_std = compute_local_std(edited_image, patch_size=64)
    # Remove pre-existing adaptation
    # normalized_noise = normalize_noise_global(normalized_noise)

    for ch in range(edited_image.shape[2]):
        intensity = edited_image[:, :, ch]  
        # expected_std = noise_func(intensity, *fitted_curves[ch])  # Noise variance in 0-1 range
        expected_std = noise_func(luminance, *fitted_curves[ch])
        # adaptive_scaling = 0.9 * expected_std + 0.1 * local_std[:, :, ch]
        
        adapted_noise[:, :, ch] = expected_std * normalized_noise[:, :, ch]
        # adapted_noise[:, :, ch] = adaptive_scaling * normalized_noise[:, :, ch]  

    # Apply mask only at the final stage
    # if mask is not None:
    #     adapted_noise *= mask  # Apply noise only in the masked region
    return adapted_noise

# 🚀 **Main Pipeline**
def dasgrain_pipeline(noisy_image, denoised_image, edited_image, mask=None):
    """Performs DASGrain noise adaptation in the 0-1 range with optional region masking."""
    
    # Step 1: Extract Noise
    noise_map = compute_noise_map(noisy_image, denoised_image)
    
    # Step 2: Compute Response Curve
    bin_centers, response_curve = compute_response_curve(noise_map, denoised_image)
    fitted_curves, noise_func = fit_response_curve(bin_centers, response_curve)
    P_R = np.poly1d(fitted_curves[0])  # Red channel polynomial
    P_G = np.poly1d(fitted_curves[1])  # Green channel polynomial
    P_B = np.poly1d(fitted_curves[2])
    # Step 3: Normalize Noise
    # normalized_noise = normalize_noise(noise_map)
    normalized_noise = normalize_with_polynomial(noise_map,[P_R,P_G,P_B])
    # tiled = tile_masked_region(normalized_noise, mask)
    # normalized_noise = preprocess_image_with_response(noise_map,response_curve)
    cv2.imwrite("noise_faulty/normalized_noise.png", np.clip((normalized_noise *25) * 255, 0, 255).astype(np.uint8))
    
    cell_map, voronoi_overlay, vor,points = generate_voronoi_pattern(normalized_noise,num_cells=500,border_extension=100)
    cv2.imwrite("noise_faulty/voronoi_overlay_normal.jpg", voronoi_overlay)
    # Scatter the Voronoi cells

    scattered_noise,translated_overlay= scatter_voronoi_cells(normalized_noise, cell_map,vor,points, max_shift=20,method='voronoi_sample')
    
    cv2.imwrite("noise_faulty/translated_voronoi_overlay_normal.jpg", translated_overlay)
    cv2.imwrite("noise_faulty/scattered_noise.jpg", np.clip((scattered_noise *25) * 255, 0, 255).astype(np.uint8))
    # Step 4: Add Adaptive Noise to Edited Image
    # plate_edited = add_adaptive_noise(edited_image, normalized_noise, fitted_curves, noise_func, mask)
    noisy_edited = add_adaptive_noise(edited_image, scattered_noise, fitted_curves, noise_func, mask)
    cv2.imwrite("noise_faulty/adapted_noise.jpg", np.clip((noisy_edited *25) * 255, 0, 255).astype(np.uint8))
    # cv2.imwrite("noise/plate_adapted_noise.png", np.clip((plate_edited *25) * 255, 0, 255).astype(np.uint8))
    final_noisy_edited = np.clip((edited_image + noisy_edited*mask + (1-mask)*noise_map) , 0.0, 1.0)
    # final_noisy_edited = np.clip((edited_image + plate_edited*mask + (1-mask)*noise_map) , 0.0, 1.0)
    # Save the output
    cv2.imwrite("noise_faulty/result.jpg", (final_noisy_edited * 255).astype(np.uint8))
    # result = np.zeros(final_noisy_edited.shape).astype('float32')
    # result = noisy_edited*mask + (1-mask)*noise_map
    # cv2.imwrite("noise/final_noisy.png", (result * 255).astype(np.uint8))
    # Plot the response curve
    # plt.figure(figsize=(6, 4))
    # for ch, color in enumerate(["Red", "Green", "Blue"]):
    #     plt.plot(bin_centers, response_curve[:, ch], 'o-', label=f"{color} Channel")
    # plt.xlabel("Intensity Level (0-1)")
    # plt.ylabel("Noise Standard Deviation")
    # plt.title("Noise Response Curve")
    # plt.legend()
    # plt.grid()
    # plt.savefig("noise_smug/response_curve.png")
    # plt.show()

    return final_noisy_edited

# **Example Usage**
if __name__ == "__main__":
    noisy_image = cv2.imread("noise_transfer/noisy_vmd.png").astype(np.float32) / 255.0  # Read and normalize
    denoised_image = cv2.imread("noise_transfer/denoised_vmd.png").astype(np.float32) / 255.0
    edited_image = cv2.imread("noise_transfer/denoised_vmd.png").astype(np.float32) / 255.0
    mask = cv2.imread("noise_faulty/neat_mask1.jpg").astype(np.float32) / 255.0  # Ensure mask is also in 0-1 range

    final_result = dasgrain_pipeline(noisy_image, denoised_image, edited_image, mask)