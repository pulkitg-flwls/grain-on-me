import numpy as np
import cv2
from scipy import ndimage, signal
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def extract_noise_pattern(noisy_image, denoised_image):
    """
    Extract the noise pattern from a noisy image by comparing with its denoised version
    
    Args:
        noisy_image (np.ndarray): Original noisy image, float32 in 0-1 range
        denoised_image (np.ndarray): Denoised version of the same image, float32 in 0-1 range
        
    Returns:
        np.ndarray: Extracted noise pattern, float32 in -1 to 1 range
    """
    # Simple subtraction to get the noise
    noise = noisy_image - denoised_image
    
    # Normalize to a reasonable range (centered at 0)
    for c in range(noise.shape[2]):
        channel_std = max(np.std(noise[:,:,c]), 0.001)  # Avoid division by zero
        noise[:,:,c] = noise[:,:,c] / (channel_std * 3)  # Scale to roughly -1 to 1 range
    
    noise = np.clip(noise, -1, 1)
    
    return noise


def analyze_noise_characteristics(noise_pattern):
    """
    Analyze noise pattern to extract key characteristics
    
    Args:
        noise_pattern (np.ndarray): Extracted noise, float32 in -1 to 1 range
        
    Returns:
        dict: Characteristics like power spectrum, intensity, channel correlation
    """
    height, width, channels = noise_pattern.shape
    characteristics = {}
    
    # Average intensity per channel
    characteristics['intensity'] = np.mean(np.abs(noise_pattern), axis=(0, 1))
    
    # Channel correlation matrix
    flat_noise = noise_pattern.reshape(-1, channels)
    characteristics['correlation_matrix'] = np.corrcoef(flat_noise.T)
    
    # Power spectrum (using just luminance for simplicity)
    luminance = 0.2126 * noise_pattern[:,:,0] + 0.7152 * noise_pattern[:,:,1] + 0.0722 * noise_pattern[:,:,2]
    fft = np.fft.fft2(luminance)
    fft_shift = np.fft.fftshift(fft)
    power_spectrum = np.abs(fft_shift)**2
    
    # Normalize and log-scale for easier analysis
    power_spectrum = np.log(power_spectrum + 1)
    characteristics['power_spectrum'] = power_spectrum
    
    # Estimate grain size by analyzing frequency distribution
    # Higher frequency content = smaller grain, lower frequency = larger grain
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.fftfreq(width)[None, :]
    freq_magnitude = np.sqrt(freq_y**2 + freq_x**2)
    
    # Weight the frequencies by their power
    weighted_freq = freq_magnitude * power_spectrum
    avg_freq = np.sum(weighted_freq) / np.sum(power_spectrum)
    
    # Convert to grain size (inversely proportional to frequency)
    characteristics['estimated_grain_size'] = 1.0 / (avg_freq * 100)
    
    return characteristics


def generate_matching_grain(height, width, noise_chars, seed=None):
    """
    Generate synthetic grain that matches extracted characteristics
    
    Args:
        height, width: Dimensions of output grain
        noise_chars: Noise characteristics from analyze_noise_characteristics
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated grain pattern matching input characteristics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate base noise for each channel
    grain = np.random.normal(0, 1, (height, width, 3)).astype(np.float32)
    
    # Filter noise to match power spectrum
    grain_size = noise_chars['estimated_grain_size']
    sigma = grain_size / 2.0
    
    # Apply filtering to match estimated grain size
    for c in range(3):
        grain[:,:,c] = ndimage.gaussian_filter(grain[:,:,c], sigma)
    
    # Apply correlation between channels to match original noise
    correlation = noise_chars['correlation_matrix']
    
    # Use Cholesky decomposition to correlate the channels
    L = np.linalg.cholesky(correlation)
    
    # Reshape grain for matrix multiplication
    flat_grain = grain.reshape(-1, 3)
    
    # Apply correlation transform
    correlated_grain = np.dot(flat_grain, L.T)
    
    # Reshape back to image dimensions
    grain = correlated_grain.reshape(height, width, 3)
    
    # Scale each channel to match original intensity
    for c in range(3):
        # Normalize
        current_intensity = np.mean(np.abs(grain[:,:,c]))
        if current_intensity > 0:
            grain[:,:,c] = grain[:,:,c] * (noise_chars['intensity'][c] / current_intensity)
    
    return grain


def regrain_image(denoised_image, grain_pattern, intensity=1.0, blend_mode='overlay'):
    """
    Apply extracted or generated grain to denoised image
    
    Args:
        denoised_image: Clean image to regrain
        grain_pattern: Grain to apply
        intensity: Overall grain strength multiplier
        blend_mode: How to blend grain with image
        
    Returns:
        np.ndarray: Regrained image
    """
    # Scale grain by intensity
    scaled_grain = grain_pattern * intensity
    
    if blend_mode == 'add':
        # Simple additive grain
        result = denoised_image + scaled_grain
        result = np.clip(result, 0, 1)
        
    elif blend_mode == 'overlay':
        # Overlay blend mode (standard in Nuke)
        # Convert grain from -1...1 to 0...1 range for blending
        grain_01 = scaled_grain * 0.5 + 0.5
        
        # Apply overlay blend mode
        bright_mask = denoised_image >= 0.5
        
        result = np.zeros_like(denoised_image)
        result[~bright_mask] = 2 * denoised_image[~bright_mask] * grain_01[~bright_mask]
        result[bright_mask] = 1 - 2 * (1 - denoised_image[bright_mask]) * (1 - grain_01[bright_mask])
        result = np.clip(result, 0, 1)
        
    elif blend_mode == 'soft_light':
        # Soft Light blend (more subtle than overlay)
        # Convert grain from -1...1 to 0...1 range for blending
        grain_01 = scaled_grain * 0.5 + 0.5
        
        result = np.zeros_like(denoised_image)
        for c in range(3):
            result[:,:,c] = ((1 - 2 * grain_01[:,:,c]) * denoised_image[:,:,c]**2 + 
                            2 * grain_01[:,:,c] * denoised_image[:,:,c])
        result = np.clip(result, 0, 1)
    
    else:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")
    
    return result


def process_reference_based(noisy_path, denoised_path, output_path, intensity=1.0, blend_mode='overlay'):
    """
    Process a single image using reference-based grain matching
    
    Args:
        noisy_path: Path to original noisy image
        denoised_path: Path to denoised version
        output_path: Where to save result
        intensity: Grain intensity multiplier
        blend_mode: Grain blend mode
    """
    # Read input images
    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
    denoised_img = cv2.imread(denoised_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to float32 and 0-1 range
    if noisy_img.dtype == np.uint8:
        noisy_img = noisy_img.astype(np.float32) / 255.0
        denoised_img = denoised_img.astype(np.float32) / 255.0
    elif noisy_img.dtype == np.uint16:
        noisy_img = noisy_img.astype(np.float32) / 65535.0
        denoised_img = denoised_img.astype(np.float32) / 65535.0
    
    # Convert BGR to RGB if needed
    if len(noisy_img.shape) == 3 and noisy_img.shape[2] == 3:
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
    
    # Extract noise pattern
    noise = extract_noise_pattern(noisy_img, denoised_img)
    
    # Analyze noise characteristics
    noise_chars = analyze_noise_characteristics(noise)
    
    # Generate matching grain (optionally, you could use the extracted noise directly)
    height, width = denoised_img.shape[:2]
    grain = generate_matching_grain(height, width, noise_chars)
    
    # Apply grain to denoised image
    result = regrain_image(denoised_img, grain, intensity, blend_mode)
    
    # Convert back to BGR for saving
    if len(result.shape) == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Convert to appropriate bit depth
    if output_path.lower().endswith(('.exr', '.hdr')):
        # Save as EXR/HDR
        cv2.imwrite(output_path, result)
    elif output_path.lower().endswith('.png'):
        # Save as 16-bit PNG
        result = np.clip(result * 65535.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path, result)
    else:
        # Save as 8-bit image
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, result)


def process_sequence_with_reference(noisy_dir, denoised_dir, output_dir, 
                                   file_pattern='*.exr', intensity=1.0, 
                                   blend_mode='overlay', num_workers=None):
    """
    Process a sequence of images using reference-based grain matching
    
    Args:
        noisy_dir: Directory with original noisy frames
        denoised_dir: Directory with denoised frames
        output_dir: Directory to save results
        file_pattern: File pattern to match
        intensity: Grain intensity multiplier
        blend_mode: Grain blend mode
        num_workers: Number of parallel workers
    """
    import glob
    
    # Set default number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input files from denoised directory
    denoised_files = sorted(glob.glob(os.path.join(denoised_dir, file_pattern)))
    
    if not denoised_files:
        print(f"No files found matching pattern {file_pattern} in {denoised_dir}")
        return
    
    print(f"Processing {len(denoised_files)} frames...")
    
    # Function to process a single frame pair
    def process_frame_pair(args):
        idx, denoised_file = args
        
        # Determine corresponding noisy file
        filename = os.path.basename(denoised_file)
        noisy_file = os.path.join(noisy_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        # Check if noisy reference exists
        if not os.path.exists(noisy_file):
            print(f"Warning: No matching noisy file for {filename}")
            return idx, None
        
        # Process the frame pair
        process_reference_based(
            noisy_file,
            denoised_file,
            output_file,
            intensity=intensity,
            blend_mode=blend_mode
        )
        
        return idx, filename
    
    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_frame_pair, enumerate(denoised_files)))
    
    print(f"Completed processing {len(results)} frames")


def extract_and_save_noise_profile(noisy_path, denoised_path, profile_path):
    """
    Extract noise characteristics and save as a profile for later use
    
    Args:
        noisy_path: Path to original noisy image
        denoised_path: Path to denoised version
        profile_path: Where to save the noise profile
    """
    # Read input images
    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
    denoised_img = cv2.imread(denoised_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to float32 and 0-1 range
    if noisy_img.dtype == np.uint8:
        noisy_img = noisy_img.astype(np.float32) / 255.0
        denoised_img = denoised_img.astype(np.float32) / 255.0
    elif noisy_img.dtype == np.uint16:
        noisy_img = noisy_img.astype(np.float32) / 65535.0
        denoised_img = denoised_img.astype(np.float32) / 65535.0
    
    # Convert BGR to RGB if needed
    if len(noisy_img.shape) == 3 and noisy_img.shape[2] == 3:
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
    
    # Extract noise pattern
    noise = extract_noise_pattern(noisy_img, denoised_img)
    
    # Analyze noise characteristics
    noise_chars = analyze_noise_characteristics(noise)
    
    # Save profile
    np.savez(profile_path, 
             intensity=noise_chars['intensity'],
             correlation_matrix=noise_chars['correlation_matrix'],
             estimated_grain_size=noise_chars['estimated_grain_size'])
    
    print(f"Noise profile saved to {profile_path}")
    print(f"Estimated grain size: {noise_chars['estimated_grain_size']}")
    print(f"Channel intensities: {noise_chars['intensity']}")
    
    return noise_chars


def apply_noise_profile(denoised_path, output_path, profile_path, intensity=1.0, blend_mode='overlay', seed=None):
    """
    Apply a saved noise profile to a denoised image
    
    Args:
        denoised_path: Path to denoised image
        output_path: Where to save result
        profile_path: Path to saved noise profile
        intensity: Grain intensity multiplier
        blend_mode: Grain blend mode
        seed: Random seed for reproducibility
    """
    # Load noise profile
    profile = np.load(profile_path)
    noise_chars = {
        'intensity': profile['intensity'],
        'correlation_matrix': profile['correlation_matrix'],
        'estimated_grain_size': float(profile['estimated_grain_size'])
    }
    
    # Read denoised image
    denoised_img = cv2.imread(denoised_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to float32 and 0-1 range
    if denoised_img.dtype == np.uint8:
        denoised_img = denoised_img.astype(np.float32) / 255.0
    elif denoised_img.dtype == np.uint16:
        denoised_img = denoised_img.astype(np.float32) / 65535.0
    
    # Convert BGR to RGB if needed
    if len(denoised_img.shape) == 3 and denoised_img.shape[2] == 3:
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
    
    # Generate matching grain
    height, width = denoised_img.shape[:2]
    grain = generate_matching_grain(height, width, noise_chars, seed=seed)
    
    # Apply grain to denoised image
    result = regrain_image(denoised_img, grain, intensity, blend_mode)
    
    # Convert back to BGR for saving
    if len(result.shape) == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Convert to appropriate bit depth
    if output_path.lower().endswith(('.exr', '.hdr')):
        # Save as EXR/HDR
        cv2.imwrite(output_path, result)
    elif output_path.lower().endswith('.png'):
        # Save as 16-bit PNG
        result = np.clip(result * 65535.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path, result)
    else:
        # Save as 8-bit image
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, result)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply reference-based film grain")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Command to process a single image
    process_parser = subparsers.add_parser("process", help="Process single image with reference")
    process_parser.add_argument("noisy", help="Path to original noisy image")
    process_parser.add_argument("denoised", help="Path to denoised image")
    process_parser.add_argument("output", help="Path to save output image")
    process_parser.add_argument("--intensity", type=float, default=1.0, help="Grain intensity multiplier")
    process_parser.add_argument("--blend", choices=["add", "overlay", "soft_light"], 
                             default="overlay", help="Grain blend mode")
    
    # Command to process a sequence
    sequence_parser = subparsers.add_parser("sequence", help="Process image sequence")
    sequence_parser.add_argument("noisy_dir", help="Directory with original noisy frames")
    sequence_parser.add_argument("denoised_dir", help="Directory with denoised frames")
    sequence_parser.add_argument("output_dir", help="Directory to save output frames")
    sequence_parser.add_argument("--pattern", default="*.exr", help="File pattern (e.g. *.exr)")
    sequence_parser.add_argument("--intensity", type=float, default=1.0, help="Grain intensity multiplier")
    sequence_parser.add_argument("--blend", choices=["add", "overlay", "soft_light"], 
                              default="overlay", help="Grain blend mode")
    sequence_parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    
    # Command to extract and save a noise profile
    extract_parser = subparsers.add_parser("extract", help="Extract noise profile")
    extract_parser.add_argument("noisy", help="Path to original noisy image")
    extract_parser.add_argument("denoised", help="Path to denoised image")
    extract_parser.add_argument("profile", help="Path to save noise profile")
    
    # Command to apply a saved noise profile
    apply_parser = subparsers.add_parser("apply", help="Apply saved noise profile")
    apply_parser.add_argument("denoised", help="Path to denoised image")
    apply_parser.add_argument("output", help="Path to save output image")
    apply_parser.add_argument("profile", help="Path to noise profile")
    apply_parser.add_argument("--intensity", type=float, default=1.0, help="Grain intensity multiplier")
    apply_parser.add_argument("--blend", choices=["add", "overlay", "soft_light"], 
                           default="overlay", help="Grain blend mode")
    apply_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_reference_based(
            args.noisy,
            args.denoised,
            args.output,
            intensity=args.intensity,
            blend_mode=args.blend
        )
    
    elif args.command == "sequence":
        process_sequence_with_reference(
            args.noisy_dir,
            args.denoised_dir,
            args.output_dir,
            file_pattern=args.pattern,
            intensity=args.intensity,
            blend_mode=args.blend,
            num_workers=args.workers
        )
    
    elif args.command == "extract":
        extract_and_save_noise_profile(
            args.noisy,
            args.denoised,
            args.profile
        )
    
    elif args.command == "apply":
        apply_noise_profile(
            args.denoised,
            args.output,
            args.profile,
            intensity=args.intensity,
            blend_mode=args.blend,
            seed=args.seed
        )
    
    else:
        parser.print_help()