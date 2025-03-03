from scipy.interpolate import griddata
from functools import lru_cache
from scipy import ndimage
import numpy as np
import cv2



def estimate_noise_profile(frame):
    """
    Estimate noise level using luminance channel with optimized scipy-based processing.
    
    Parameters:
        frame: Input frame in BGR format
        
    Returns:
        noise_map: Spatial map of estimated noise levels
        params: Dictionary of suggested bilateral filter parameters for each scale
    """
    # Convert to YCrCb and extract luminance channel
    if frame.ndim == 3:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        luminance = ycrcb[..., 0].astype(np.float32)
    else:
        luminance = frame.copy().astype(np.float32)  # Already single channel
    
    # 1. Find flat regions (low gradient) for noise estimation
    # Gradient magnitude using Sobel operators
    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    # Threshold to find flat regions (lower percentile of gradient magnitude)
    flat_threshold = np.percentile(gradient_mag, 20)  # Bottom 20% are considered flat
    flat_mask = gradient_mag < flat_threshold
    
    # 2. Estimate noise using optimized scipy-based method
    noise_map = estimate_noise_with_scipy(luminance, flat_mask, patch_size=7)
    
    # 3. Determine bilateral filter parameters based on noise levels
    noise_min = np.percentile(noise_map, 10)
    noise_max = np.percentile(noise_map, 90)
    noise_median = np.median(noise_map)
    noise_mean = np.mean(noise_map)
    
    # Scale parameters based on noise level
    # These formulas adapt bilateral filter parameters to the estimated noise
    params = {
        'fine_scale': {
            'd': 5,
            'sigma_color': max(10, min(40, noise_median * 2.5)),
            'sigma_space': max(15, min(40, 15 + noise_median * 1.5))
        },
        'medium_scale': {
            'd': 5,
            'sigma_color': max(20, min(60, noise_median * 3.5)),
            'sigma_space': max(30, min(70, 30 + noise_median * 2.0))
        },
        'coarse_scale': {
            'd': 5,
            'sigma_color': max(40, min(90, noise_median * 4.5)),
            'sigma_space': max(50, min(100, 50 + noise_median * 2.5))
        }
    }
    
    print(f"Estimated noise parameters: Min={noise_min:.2f}, Max={noise_max:.2f}, "
          f"Median={noise_median:.2f}, Mean={noise_mean:.2f}")
    print(f"Fine scale: color={params['fine_scale']['sigma_color']:.2f}, "
          f"space={params['fine_scale']['sigma_space']:.2f}")
    print(f"Medium scale: color={params['medium_scale']['sigma_color']:.2f}, "
          f"space={params['medium_scale']['sigma_space']:.2f}")
    print(f"Coarse scale: color={params['coarse_scale']['sigma_color']:.2f}, "
          f"space={params['coarse_scale']['sigma_space']:.2f}")
    
    return noise_map, params


def estimate_noise_with_scipy(luminance, flat_mask, patch_size=7):
    """
    Estimate noise using scipy's efficient local statistic functions.
    
    Parameters:
        luminance: Luminance channel of the image
        flat_mask: Binary mask indicating flat regions (True for flat)
        patch_size: Size of patches for local noise estimation
        
    Returns:
        noise_map: Spatial map of estimated noise levels
    """
    
    
    # Calculate local standard deviation
    # First calculate local mean and local squared mean
    # Zero out non-flat regions
    flat_luminance = luminance.copy()
    flat_luminance[~flat_mask] = 0
    
    # Count flat pixels in each patch
    flat_count = ndimage.uniform_filter(flat_mask.astype(np.float32), size=patch_size)
    valid_patches = flat_count > 0.3  # At least 30% of patch must be flat
    
    # If not enough valid patches, use global approach
    if np.sum(valid_patches) < 0.1 * luminance.size:
        flat_pixels = luminance[flat_mask]
        if len(flat_pixels) > 0:
            global_noise = np.std(flat_pixels)
        else:
            global_noise = 5.0
        return np.ones_like(luminance, dtype=np.float32) * global_noise
    
    # Calculate sum and sum of squares in flat regions
    flat_sum = ndimage.uniform_filter(flat_luminance, size=patch_size)
    flat_sum_sq = ndimage.uniform_filter(flat_luminance**2, size=patch_size)
    
    # Adjust for number of flat pixels in each patch
    flat_count = np.maximum(flat_count, 1e-6)  # Avoid division by zero
    flat_mean = flat_sum / flat_count
    flat_mean_sq = flat_sum_sq / flat_count
    
    # Compute variance and standard deviation
    flat_var = np.maximum(0, flat_mean_sq - flat_mean**2)
    flat_std = np.sqrt(flat_var)
    
    # Only use values from valid patches
    noise_map = np.zeros_like(luminance, dtype=np.float32)
    noise_map[valid_patches] = flat_std[valid_patches]
    
    # Fill gaps using interpolation
    # Get coordinates of valid estimates
    y_valid, x_valid = np.where(valid_patches)
    valid_values = noise_map[y_valid, x_valid]
    
    # Remove zeros and very low values
    non_zero = valid_values > 0.1
    if np.any(non_zero):
        y_valid = y_valid[non_zero]
        x_valid = x_valid[non_zero]
        valid_values = valid_values[non_zero]
    
    # If no valid values remain, return default
    if len(valid_values) == 0:
        return np.ones_like(luminance, dtype=np.float32) * 5.0
    
    # Create grid for interpolation
    y_grid, x_grid = np.mgrid[0:noise_map.shape[0], 0:noise_map.shape[1]]
    
    # Interpolate missing values
    noise_map_filled = griddata(
        (y_valid, x_valid), 
        valid_values, 
        (y_grid, x_grid), 
        method='nearest', 
        fill_value=np.median(valid_values)
    )
    
    # Smooth the result for more stable parameters
    noise_map_smooth = ndimage.gaussian_filter(noise_map_filled, sigma=5.0)
    
    return noise_map_smooth


def adaptive_bilateral_blend(ref_frame, warped_neighbors, weights, occlusion_mask, edge_mask=None):
    """
    Blend frames with automatically estimated noise profile.
    
    Parameters:
      ref_frame: numpy array of shape (H, W, 3), the reference image.
      warped_neighbors: list of numpy arrays of shape (H, W, 3) for each neighbor.
      weights: numpy array of shape (N, H, W) containing per-pixel weights for each neighbor.
      occlusion_mask: numpy array of shape (H, W) with binary values (0: occluded, 1: valid).
      edge_mask: Optional mask to preserve edges, shape (H, W, 1).
      
    Returns:
      final_output: Blended image of shape (H, W, 3) with multi-scale frequency selective denoising.
    """
    # Estimate noise profile and parameters
    noise_map, bilateral_params = estimate_noise_profile(ref_frame)
    
    # Multi-scale decomposition parameters from estimated noise
    scale_params = [
        bilateral_params['fine_scale'],     # Fine scale (highest frequencies)
        bilateral_params['medium_scale'],   # Medium scale
        bilateral_params['coarse_scale']    # Coarse scale (lowest frequencies)
    ]
    
    # Expand weights for broadcasting
    weights_expanded = weights[..., None]  # Shape: (N, H, W, 1)
    
    # Initialize frequency bands for reference frame
    ref_bands = []
    ref_residual = ref_frame.astype(np.float32)
    
    # Decompose reference frame into multiple frequency bands
    for scale_idx, params in enumerate(scale_params):
        # Extract parameters
        d = params['d']
        sigma_color = params['sigma_color']
        sigma_space = params['sigma_space']
        
        # Apply bilateral filter
        current_band = np.zeros_like(ref_frame, dtype=np.float32)
        for c in range(3):
            current_band[..., c] = cv2.bilateralFilter(
                ref_residual[..., c].astype(np.float32), d, sigma_color, sigma_space
            )
        
        # Extract current band details (difference between residual and filtered)
        band_details = ref_residual - current_band
        ref_bands.append(band_details)
        
        # Update residual for next scale
        ref_residual = current_band
    
    # Last residual is the coarsest low-frequency component
    ref_bands.append(ref_residual)
    
    # Process warped neighbors to extract frequency bands
    neighbor_bands = []
    
    for neighbor in warped_neighbors:
        bands = []
        residual = neighbor.astype(np.float32)
        
        # Extract bands
        for scale_idx, params in enumerate(scale_params):
            # Extract parameters
            d = params['d']
            sigma_color = params['sigma_color']
            sigma_space = params['sigma_space']
            
            # Apply bilateral filter
            current_band = np.zeros_like(neighbor, dtype=np.float32)
            for c in range(3):
                current_band[..., c] = cv2.bilateralFilter(
                    residual[..., c].astype(np.float32), d, sigma_color, sigma_space
                )
            
            # Extract current band details
            band_details = residual - current_band
            bands.append(band_details)
            
            # Update residual
            residual = current_band
        
        # Add coarsest band
        bands.append(residual)
        
        neighbor_bands.append(bands)
    
    # Number of frequency bands (including residual)
    num_bands = len(ref_bands)
    
    # Process each frequency band with appropriate strategy
    processed_bands = []
    
    for band_idx in range(num_bands):
        # Get reference band
        ref_band = ref_bands[band_idx]
        
        # Get same band from neighbors
        neighbor_band = [n_bands[band_idx] for n_bands in neighbor_bands]
        
        # Determine processing strategy based on band index
        if band_idx == num_bands - 1:
            # Coarsest band (lowest frequencies) - conservative blending
            processed_band = blend_low_frequency_band(
                ref_band, neighbor_band, weights_expanded * 0.7, occlusion_mask, edge_mask
            )
        else:
            # Detail bands - compute consistency and apply adaptive blending
            # Higher bands (higher frequencies) get more aggressive processing
            consistency = compute_band_consistency(ref_band, neighbor_band, occlusion_mask)
            
            # Adjust consistency threshold based on band (higher threshold for higher frequencies)
            lower_threshold = 0.25 - 0.05 * band_idx  # 0.25, 0.2, 0.15 for 3 bands
            upper_threshold = 0.65 - 0.05 * band_idx  # 0.65, 0.6, 0.55 for 3 bands
            transition_range = upper_threshold - lower_threshold
            
            detail_mask = np.clip((consistency - lower_threshold) / transition_range, 0, 1)[..., None]
            
            # Higher bands (index 0 = highest freq) get more aggressive noise reduction
            denoising_strength = 0.5 + 0.2 * (num_bands - 2 - band_idx)  # 0.9, 0.7, 0.5 for 3 bands
            
            processed_band = blend_detail_band(
                ref_band, neighbor_band, weights_expanded, occlusion_mask,
                detail_mask, edge_mask, denoising_strength
            )
        
        processed_bands.append(processed_band)
    
    # Recombine frequency bands
    final_output = np.zeros_like(ref_frame, dtype=np.float32)
    for band in processed_bands:
        final_output += band
    
    # Optional: Apply subtle enhancement to counteract any residual blurriness
    if edge_mask is not None:
        # Edge-aware enhancement only at strong edges
        edge_enhanced = final_output + edge_mask * (final_output - cv2.GaussianBlur(final_output, (3, 3), 0)) * 0.5
        final_output = edge_enhanced
    
    return np.clip(final_output, 0, 255).astype(np.uint8)

def decompose_image(image, scale_params):
    """
    Decompose image into multiple frequency bands using cascaded bilateral filters.
    Optimized for memory efficiency by reusing arrays.
    """
    # Initialize
    bands = []
    residual = image.astype(np.float32)
    
    # Extract bands
    for params in scale_params:
        d = params['d']
        sigma_color = params['sigma_color']
        sigma_space = params['sigma_space']
        
        # Apply bilateral filter efficiently
        filtered = np.zeros_like(residual)
        for c in range(3):
            filtered[..., c] = cv2.bilateralFilter(
                residual[..., c].astype(np.float32), 
                d, sigma_color, sigma_space
            )
        
        # Extract band details (current residual - filtered)
        band_details = residual - filtered
        bands.append(band_details)
        
        # Update residual
        residual = filtered
    
    # Add coarsest band
    bands.append(residual)
    
    return bands


def blend_low_frequency_band(ref_band, neighbor_bands, weights, occlusion_mask, edge_mask=None):
    """Optimized blending for lowest frequency band."""
    # Start with reference
    blended_band = ref_band.copy() * 0.5
    total_weight = np.ones_like(ref_band[..., 0:1]) * 0.5
    
    # Edge-aware weight adjustment
    if edge_mask is not None:
        effective_weights = [w * (1 - edge_mask * 0.8) for w in weights]
    else:
        effective_weights = weights
    
    # Valid regions mask - compute once
    valid_mask = (1 - occlusion_mask)[..., None]
    
    # Vectorized blending for all neighbors
    for neighbor_band, w in zip(neighbor_bands, effective_weights):
        weight_map = w * (1 - valid_mask)
        blended_band += neighbor_band * weight_map
        total_weight += weight_map
    
    # Normalize
    total_weight = np.maximum(total_weight, 1e-6)
    blended_band /= total_weight
    
    # Handle occlusions
    occlusion_mask_expanded = occlusion_mask[..., None]
    return np.where(occlusion_mask_expanded == 1, blended_band, ref_band)


def blend_detail_band(ref_band, neighbor_bands, weights, occlusion_mask, 
                      detail_mask, edge_mask=None, denoising_strength=0.7):
    """Optimized blending for detail bands."""
    # Initialize - preallocate arrays
    blended_band = np.zeros_like(ref_band, dtype=np.float32)
    total_weight = np.zeros_like(ref_band[..., 0:1], dtype=np.float32)
    
    # Edge-aware weight adjustment
    if edge_mask is not None:
        effective_weights = [w * (1 - edge_mask * 0.9) for w in weights]
    else:
        effective_weights = weights
    
    # Precompute valid mask
    valid_mask = (1 - occlusion_mask)[..., None]
    
    # Vectorized blending
    for neighbor_band, w in zip(neighbor_bands, effective_weights):
        # Adjust weights based on detail importance
        adjusted_weight = w * (1 - valid_mask) * (1 - detail_mask * 0.8)
        blended_band += neighbor_band * adjusted_weight
        total_weight += adjusted_weight
    
    # Normalize with vectorized operations
    valid_pixels = (total_weight > 1e-6)
    blended_band = np.where(valid_pixels, blended_band / total_weight, 0)
    
    # Blend with reference based on detail mask
    detail_strength = 1.0 + (1.0 - denoising_strength) * 0.3
    final_band = ref_band * detail_mask * detail_strength + blended_band * (1 - detail_mask * denoising_strength)
    
    # Handle occlusions
    occlusion_mask_expanded = occlusion_mask[..., None]
    return np.where(occlusion_mask_expanded == 1, final_band, ref_band)


def compute_band_consistency(ref_band, neighbor_bands, occlusion_mask):
    """Compute frequency band consistency across frames - vectorized implementation."""
    # Initialize arrays
    consistency_sum = np.zeros(ref_band.shape[:2], dtype=np.float32)
    weight_sum = np.zeros(ref_band.shape[:2], dtype=np.float32)
    
    # Precompute valid mask
    valid_mask = 1 - occlusion_mask
    
    # For each neighbor
    for neighbor_band in neighbor_bands:
        # Compute per-channel agreement
        channel_agreement = np.zeros(ref_band.shape[:2], dtype=np.float32)
        
        for c in range(3):
            # Magnitude-based agreement
            ref_abs = np.abs(ref_band[..., c])
            neigh_abs = np.abs(neighbor_band[..., c])
            
            # Fast local standard deviation (5x5 window)
            ref_std = local_std(ref_band[..., c])
            neigh_std = local_std(neighbor_band[..., c])
            
            # Weight by local contrast - vectorized operations
            local_importance = np.maximum(ref_std, neigh_std) + 1e-6
            max_importance = local_importance.max() + 1e-6
            
            # Normalized agreement
            max_val = np.maximum(ref_abs, neigh_abs) + 1e-6
            agreement = 1 - np.abs(ref_abs - neigh_abs) / max_val
            
            # Weight by importance
            channel_agreement += (agreement * (local_importance / max_importance)) / 3
        
        # Accumulate
        consistency_sum += channel_agreement * valid_mask
        weight_sum += valid_mask
    
    # Normalize
    weight_sum = np.maximum(weight_sum, 1.0)
    return consistency_sum / weight_sum


@lru_cache(maxsize=8)
def get_uniform_kernel(window_size):
    """Create and cache uniform kernel for filter operations."""
    return np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)


def local_std(image, window_size=5):
    """Fast local standard deviation using separable convolutions."""
    # Get or create uniform kernel
    kernel = get_uniform_kernel(window_size)
    
    # Compute local mean (separable filter)
    local_mean = cv2.filter2D(image, -1, kernel)
    
    # Compute local squared mean (separable filter)
    local_sqr_mean = cv2.filter2D(image**2, -1, kernel)
    
    # Compute local variance and standard deviation
    local_var = np.maximum(0, local_sqr_mean - local_mean**2)
    return np.sqrt(local_var)

def bilateral_frequency_blend(ref_frame, warped_neighbors, weights, occlusion_mask, edge_mask=None):
    """
    Blend frames using bilateral filter for frequency separation with motion consistency.
    
    Parameters:
      ref_frame: numpy array of shape (H, W, 3), the reference image.
      warped_neighbors: list of numpy arrays of shape (H, W, 3) for each neighbor.
      weights: numpy array of shape (N, H, W) containing per-pixel weights for each neighbor.
      occlusion_mask: numpy array of shape (H, W) with binary values (0: occluded, 1: valid).
      edge_mask: Optional mask to preserve edges, shape (H, W, 1).
      
    Returns:
      final_output: Blended image of shape (H, W, 3) with frequency-selective denoising.
    """
    # Parameters for bilateral filtering
    d = 9  # Filter diameter
    sigma_color = 50  # Range domain standard deviation
    sigma_space = 75  # Spatial domain standard deviation
    
    # Expand weights for broadcasting
    weights_expanded = weights[..., None]  # Shape: (N, H, W, 1)
    
    # Extract low frequency component from reference frame using bilateral filter
    ref_low = np.zeros_like(ref_frame, dtype=np.float32)
    for c in range(3):
        ref_low[..., c] = cv2.bilateralFilter(ref_frame[..., c].astype(np.float32), 
                                             d, sigma_color, sigma_space)
    
    # Extract high frequency component (detail layer)
    ref_high = ref_frame.astype(np.float32) - ref_low
    
    # Process each neighbor to extract low and high frequencies
    neighbors_low = []
    neighbors_high = []
    
    for neighbor in warped_neighbors:
        # Extract low frequency using bilateral filter
        neighbor_low = np.zeros_like(neighbor, dtype=np.float32)
        for c in range(3):
            neighbor_low[..., c] = cv2.bilateralFilter(neighbor[..., c].astype(np.float32), 
                                                     d, sigma_color, sigma_space)
        
        # Extract high frequency
        neighbor_high = neighbor.astype(np.float32) - neighbor_low
        
        neighbors_low.append(neighbor_low)
        neighbors_high.append(neighbor_high)
    
    # Compute consistency of high frequencies across frames (after warping)
    # High consistency indicates true detail, low consistency indicates noise
    consistency = compute_high_freq_consistency(ref_high, neighbors_high, occlusion_mask)
    
    # Adjust weights for frequency-selective blending
    if edge_mask is not None:
        # Preserve edges in both frequency bands
        edge_weight_factor = 1 - edge_mask[None, :, :, 0]
        low_freq_weights = weights_expanded * edge_weight_factor[..., None]
        
        # Even more preservation for edges in high frequencies
        high_freq_weights = weights_expanded * np.minimum(edge_weight_factor, 0.7)[..., None]
    else:
        # Default weights if no edge mask
        low_freq_weights = weights_expanded * 0.8  # Conservative for structure
        high_freq_weights = weights_expanded      # Standard for high frequencies
    
    # Create detail preservation mask from consistency
    # High consistency (>0.7) = true detail to preserve
    # Low consistency (<0.3) = likely noise to remove
    detail_mask = np.clip((consistency - 0.3) / 0.4, 0, 1)[..., None]
    
    # Adjust high frequency weights to be more aggressive for low-consistency areas
    for i in range(len(high_freq_weights)):
        # Reduce influence of reference in noise areas (low consistency)
        high_freq_weights[i] = high_freq_weights[i] * (1 - detail_mask * 0.7)
    
    # Blend low frequency components (structure)
    low_freq_blended = np.zeros_like(ref_frame, dtype=np.float32)
    low_freq_total_weight = np.zeros(ref_frame.shape[:2] + (1,), dtype=np.float32)
    
    # Start with reference frame
    low_freq_blended += ref_low * 0.5  # Give reference frame a base weight
    low_freq_total_weight += 0.5
    
    # Add weighted neighbors
    for neighbor_low, w in zip(neighbors_low, low_freq_weights):
        valid_mask = (1 - occlusion_mask)[..., None]  # Valid = 0, occluded = 1
        effective_weight = w * (1 - valid_mask)
        
        low_freq_blended += neighbor_low * effective_weight
        low_freq_total_weight += effective_weight
    
    # Normalize
    low_freq_total_weight = np.clip(low_freq_total_weight, 1e-6, None)
    low_freq_blended /= low_freq_total_weight
    
    # Blend high frequency components (details and noise)
    high_freq_blended = np.zeros_like(ref_frame, dtype=np.float32)
    high_freq_total_weight = np.zeros(ref_frame.shape[:2] + (1,), dtype=np.float32)
    
    # Add weighted neighbors
    for neighbor_high, w in zip(neighbors_high, high_freq_weights):
        valid_mask = (1 - occlusion_mask)[..., None]
        effective_weight = w * (1 - valid_mask)
        
        high_freq_blended += neighbor_high * effective_weight
        high_freq_total_weight += effective_weight
    
    # Normalize
    high_freq_total_weight = np.clip(high_freq_total_weight, 1e-6, None)
    high_freq_blended /= high_freq_total_weight
    
    # For high consistency areas, preserve reference details
    # For low consistency areas, use blended result (noise reduction)
    final_high_freq = ref_high * detail_mask + high_freq_blended * (1 - detail_mask)
    
    # Handle occlusions in each frequency band
    occlusion_mask_expanded = occlusion_mask[..., None]
    low_freq_result = np.where(occlusion_mask_expanded == 1, 
                              low_freq_blended, 
                              ref_low)
    
    high_freq_result = np.where(occlusion_mask_expanded == 1,
                               final_high_freq,
                               ref_high)
    
    # Recombine frequency bands
    final_output = low_freq_result + high_freq_result
    
    return np.clip(final_output, 0, 255).astype(np.uint8)


def compute_high_freq_consistency(ref_high, neighbors_high, occlusion_mask):
    """
    Compute how consistent high frequencies are across warped frames.
    Higher values indicate true details, lower values indicate noise.
    
    Parameters:
      ref_high: High frequency component of reference frame.
      neighbors_high: List of high frequency components from warped neighbors.
      occlusion_mask: Binary mask indicating occluded pixels.
      
    Returns:
      consistency: Map of values [0,1] indicating temporal consistency.
    """
    # Initialize
    consistency_sum = np.zeros(ref_high.shape[:2], dtype=np.float32)
    weight_sum = np.zeros(ref_high.shape[:2], dtype=np.float32)
    
    # For each neighbor
    for i, neighbor_high in enumerate(neighbors_high):
        # Get valid (non-occluded) regions
        valid_mask = 1 - occlusion_mask
        
        # Compute per-channel agreement
        channel_agreement = np.zeros(ref_high.shape[:2], dtype=np.float32)
        
        for c in range(3):
            # Get absolute magnitudes
            ref_abs = np.abs(ref_high[..., c])
            neigh_abs = np.abs(neighbor_high[..., c])
            
            # Compute normalized agreement (1 = perfect match, 0 = maximum difference)
            max_val = np.maximum(ref_abs, neigh_abs) + 1e-6
            channel_agreement += (1 - np.abs(ref_abs - neigh_abs) / max_val) / 3
        
        # Accumulate weighted agreement
        consistency_sum += channel_agreement * valid_mask
        weight_sum += valid_mask
    
    # Normalize by total valid neighbors
    weight_sum = np.maximum(weight_sum, 1.0)  # Avoid division by zero
    consistency = consistency_sum / weight_sum
    
    return consistency

