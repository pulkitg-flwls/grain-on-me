import cv2
import numpy as np
from scipy import ndimage
from skimage.util import view_as_blocks


def detect_textureless_regions(image, block_size=16, threshold=0.02, fraction=0.1):
    """
    Detect regions with minimal texture/detail for optimal noise sampling
    using optimized library functions
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to analyze
    block_size : int
        Size of blocks to analyze (default: 16)
    threshold : float
        Variance threshold below which blocks are considered textureless (default: 0.02)
    fraction : float
        Fraction of the most textureless blocks to select (default: 0.1)
    
    Returns:
    --------
    numpy.ndarray
        Binary mask where 1 indicates textureless regions
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure we're working with float values
    gray = gray.astype(np.float32)
        
    # Normalize to 0-1 range if needed
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    # Pad the image if dimensions are not divisible by block_size
    pad_y = (0, block_size - height % block_size) if height % block_size != 0 else (0, 0)
    pad_x = (0, block_size - width % block_size) if width % block_size != 0 else (0, 0)
    
    if pad_y[1] > 0 or pad_x[1] > 0:
        gray = np.pad(gray, (pad_y, pad_x), mode='reflect')
    
    # Calculate number of blocks
    num_blocks_y = gray.shape[0] // block_size
    num_blocks_x = gray.shape[1] // block_size
    
    # Compute gradient magnitude using Sobel operator (very efficient)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Compute local variance (efficient way to detect texture)
    local_variance = ndimage.uniform_filter(gray**2, size=3) - ndimage.uniform_filter(gray, size=3)**2
    
    # Combine metrics as a texture indicator (lower values = less texture)
    texture_metric = gradient_magnitude + 10 * local_variance
    
    # Reshape to blocks using view_as_blocks (much faster than manual iteration)
    try:
        texture_blocks = view_as_blocks(texture_metric, (block_size, block_size))
    except ValueError:
        # If dimensions don't match (rare edge case), use resize
        new_height = num_blocks_y * block_size
        new_width = num_blocks_x * block_size
        texture_metric_resized = cv2.resize(texture_metric, (new_width, new_height))
        texture_blocks = view_as_blocks(texture_metric_resized, (block_size, block_size))
    
    # Calculate mean texture value for each block (vectorized)
    block_means = np.mean(texture_blocks, axis=(2, 3)).ravel()
    
    # Find the blocks with lowest texture metric (most textureless)
    num_selected = max(1, int(len(block_means) * fraction))
    threshold_value = np.partition(block_means, num_selected)[num_selected-1]
    
    # Create mask based on threshold
    block_mask = (block_means <= threshold_value).reshape(num_blocks_y, num_blocks_x)
    
    # Upsample the block mask to full image size
    mask = np.zeros((num_blocks_y * block_size, num_blocks_x * block_size), dtype=np.uint8)
    
    # Use broadcasting to efficiently assign blocks to mask
    mask_view = mask.reshape(num_blocks_y, block_size, num_blocks_x, block_size)
    mask_view[:, :, :, :] = block_mask[:, np.newaxis, :, np.newaxis] * 255
    
    # Crop mask to original dimensions
    mask = mask[:height, :width]
    
    return mask


def visualize_textureless_regions(image, mask, alpha=0.5):
    """
    Visualize detected textureless regions overlaid on the original image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Binary mask where 1 indicates textureless regions
    alpha : float
        Opacity of the overlay (default: 0.5)
    
    Returns:
    --------
    numpy.ndarray
        Visualization with textureless regions highlighted
    """
    # Ensure image is in the right format
    if image.dtype != np.uint8:
        display_img = (image * 255).astype(np.uint8)
    else:
        display_img = image.copy()
    
    # Convert to BGR if grayscale
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    # Create colored mask overlay (green for textureless regions)
    overlay = np.zeros_like(display_img)
    mask_3ch = np.stack([mask, mask, mask], axis=2) if len(mask.shape) == 2 else mask
    overlay[mask_3ch > 0] = [0, 255, 0]  # Green for textureless regions
    
    # Blend the images
    result = cv2.addWeighted(display_img, 1, overlay, alpha, 0)
    
    return result


def extract_textureless_noise(noisy, denoised, block_size=16, fraction=0.1):
    """
    Extract noise specifically from textureless regions for better noise profile
    
    Parameters:
    -----------
    noisy : numpy.ndarray
        Noisy original image
    denoised : numpy.ndarray
        Denoised version of the image
    block_size : int
        Size of blocks to analyze (default: 16)
    fraction : float
        Fraction of the most textureless blocks to select (default: 0.1)
    
    Returns:
    --------
    tuple
        (extracted_noise, textureless_mask)
    """
    # Detect textureless regions in the denoised image using optimized function
    mask = detect_textureless_regions(denoised, block_size, fraction=fraction)
    
    # Extract difference between noisy and denoised (noise) - vectorized
    noise = noisy.astype(np.float32) - denoised.astype(np.float32)
    
    # Create a 3-channel mask if needed
    if len(mask.shape) == 2 and len(noise.shape) == 3:
        mask3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    else:
        mask3 = mask
    
    # Convert mask to float for proper multiplication
    mask_norm = mask3.astype(np.float32) / 255.0
    
    return noise, mask_norm


def analyze_textureless_noise(noisy, denoised, block_size=16, fraction=0.1):
    """
    Analyze noise characteristics specifically from textureless regions
    
    Parameters:
    -----------
    noisy : numpy.ndarray
        Noisy original image
    denoised : numpy.ndarray
        Denoised version of the image
    block_size : int
        Size of blocks to analyze (default: 16)
    fraction : float
        Fraction of the most textureless blocks to select (default: 0.1)
    
    Returns:
    --------
    dict
        Dictionary containing noise statistics from textureless regions
    """
    # Extract noise from textureless regions
    noise, mask = extract_textureless_noise(noisy, denoised, block_size, fraction)
    
    # Calculate noise statistics
    stats = {}
    
    # Get masked noise (only from textureless regions)
    if np.any(mask > 0):  # Ensure we have at least some textureless pixels
        # Use boolean indexing for efficient masked analysis
        mask_indices = (mask > 0)
        noise_masked = noise[mask_indices]
        
        # Reshape to handle multi-channel images
        if len(noise.shape) == 3:
            noise_masked = noise_masked.reshape(-1, noise.shape[2])
        
        # Calculate basic statistics using numpy's optimized functions
        stats['mean'] = np.mean(noise_masked, axis=0)
        stats['std'] = np.std(noise_masked, axis=0)
        stats['min'] = np.min(noise_masked, axis=0)
        stats['max'] = np.max(noise_masked, axis=0)
        
        # Convert to YCrCb for luminance/chrominance analysis
        if len(noise.shape) == 3 and noise.shape[2] == 3:  # Only for RGB images
            # Add 0.5 to make noise centered around 0.5 for color conversion
            noise_offset = noise + 0.5
            noise_ycrcb = cv2.cvtColor(noise_offset, cv2.COLOR_BGR2YCrCb)
            # Remove offset
            noise_ycrcb = noise_ycrcb - 0.5
            
            # Get only textureless regions
            ycrcb_masked = noise_ycrcb[mask_indices]
            ycrcb_masked = ycrcb_masked.reshape(-1, noise_ycrcb.shape[2])
            
            stats['ycrcb_mean'] = np.mean(ycrcb_masked, axis=0)
            stats['ycrcb_std'] = np.std(ycrcb_masked, axis=0)
            
            # Calculate power spectrum - use Y channel for frequency analysis
            y_channel = noise_ycrcb[:,:,0]
            
            # Apply mask to get only textureless regions
            y_masked = y_channel.copy()
            y_masked[~mask_indices[:,:,0]] = 0  # Zero-out non-selected areas
            
            # Use scipy's FFT instead of numpy for better performance
            from scipy import fftpack
            f_transform = fftpack.fft2(y_masked)
            f_transform_shifted = fftpack.fftshift(f_transform)
            power_spectrum = np.abs(f_transform_shifted)**2
            
            # Store log power spectrum
            stats['power_spectrum'] = np.log1p(power_spectrum)
            
            # Calculate auto-correlation for texture analysis
            from scipy import signal
            auto_corr = signal.correlate2d(y_masked, y_masked, mode='same')
            stats['auto_correlation'] = auto_corr / np.max(auto_corr)
    
    return stats, mask


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Detect textureless regions for noise sampling")
    parser.add_argument("--noisy", required=True, help="Path to noisy image")
    parser.add_argument("--denoised", required=True, help="Path to denoised image")
    parser.add_argument("--block-size", type=int, default=16, help="Analysis block size")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of textureless blocks to select")
    parser.add_argument("--output-mask", help="Path to save textureless mask")
    parser.add_argument("--output-vis", help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Load images
    noisy_img = cv2.imread(args.noisy)
    denoised_img = cv2.imread(args.denoised)
    
    # Normalize to 0-1 range if needed
    if noisy_img.dtype == np.uint8:
        noisy_img = noisy_img.astype(np.float32) / 255.0
        denoised_img = denoised_img.astype(np.float32) / 255.0
    
    # Extract and analyze textureless noise
    stats, mask = analyze_textureless_noise(
        noisy_img, 
        denoised_img, 
        block_size=args.block_size,
        fraction=args.fraction
    )
    
    # Create visualization
    vis_img = visualize_textureless_regions(denoised_img, mask > 0)
    
    # Convert for matplotlib visualization
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # Display results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Noisy Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
    plt.title("Denoised Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    mask_vis = mask[:,:,0] if len(mask.shape) == 3 else mask
    plt.imshow(mask_vis, cmap='gray')
    plt.title("Textureless Regions Mask")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(vis_img_rgb)
    plt.title("Textureless Regions Visualized")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save outputs if requested
    if args.output_mask:
        mask_out = (mask * 255).astype(np.uint8)
        cv2.imwrite(args.output_mask, mask_out)
        print(f"Saved mask to {args.output_mask}")
    
    if args.output_vis:
        cv2.imwrite(args.output_vis, vis_img)
        print(f"Saved visualization to {args.output_vis}")
    
    # Print noise statistics
    print("\nNoise Statistics from Textureless Regions:")
    print(f"Mean: {stats['mean']}")
    print(f"Std Dev: {stats['std']}")
    print(f"Min/Max: {stats['min']} to {stats['max']}")
    
    if 'ycrcb_mean' in stats:
        print("\nYCrCb Analysis:")
        print(f"Y (Luminance) Mean: {stats['ycrcb_mean'][0]}")
        print(f"Y (Luminance) Std Dev: {stats['ycrcb_std'][0]}")
        print(f"Cr Mean: {stats['ycrcb_mean'][1]}")
        print(f"Cb Mean: {stats['ycrcb_mean'][2]}")