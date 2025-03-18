import numpy as np
import cv2
from scipy import ndimage
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from enum import Enum
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count


class GrainPreset(Enum):
    """Film stock presets mimicking Nuke's options"""
    KODAK_5248 = {'size': 3.0, 'intensity': 0.4, 'irregularity': 0.6, 'r_scale': 1.0, 'g_scale': 0.7, 'b_scale': 0.9}
    KODAK_5279 = {'size': 2.5, 'intensity': 0.5, 'irregularity': 0.5, 'r_scale': 1.0, 'g_scale': 0.8, 'b_scale': 0.9}
    KODAK_VISION3 = {'size': 1.8, 'intensity': 0.3, 'irregularity': 0.4, 'r_scale': 1.0, 'g_scale': 0.9, 'b_scale': 0.8}
    FUJI_3510 = {'size': 2.0, 'intensity': 0.35, 'irregularity': 0.5, 'r_scale': 1.0, 'g_scale': 0.9, 'b_scale': 0.7}
    ILFORD_HP5 = {'size': 3.5, 'intensity': 0.6, 'irregularity': 0.7, 'r_scale': 1.0, 'g_scale': 1.0, 'b_scale': 1.0}
    ARRI = {'size': 2.0, 'intensity': 0.35, 'irregularity': 0.5, 'r_scale':0.85, 'g_scale':1.0, 'b_scale':0.9}
    TUNGSTEN = {'size': 1.5, 'intensity': 0.25, 'irregularity': 0.3, 'r_scale': 0.85, 'g_scale': 0.9, 'b_scale': 1.0}
    REGA = {'size': 1.3, 'intensity': 0.6, 'irregularity': 0.8, 'r_scale': 0.85, 'g_scale': 0.9, 'b_scale': 1.0}


class BlendMode(Enum):
    """Blend modes for grain application"""
    ADD = 1
    OVERLAY = 2
    SOFT_LIGHT = 3


def generate_base_noise(height, width, grain_size, irregularity=0.5, seed=None):
    """
    Generate base noise texture for film grain
    
    Args:
        height (int): Image height
        width (int): Image width
        grain_size (float): Size of grain particles (higher = larger grains)
        irregularity (float): How irregular the grain pattern is (0-1)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Noise texture as float32 array with range -1 to 1
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate white noise
    noise = np.random.normal(0, 1, (height, width)).astype(np.float32)
    
    # Scale the kernel size based on grain_size
    # (larger grain_size = more blurring = larger grain particles)
    kernel_size = max(1, int(grain_size * 2))
    
    # Create a Gaussian kernel for the primary blur
    primary_sigma = grain_size / 2
    if primary_sigma > 0:
        noise = ndimage.gaussian_filter(noise, primary_sigma)
    
    # Add irregularity by selectively emphasizing certain frequencies
    if irregularity > 0:
        # Create a second noise pattern for irregularity
        irregular_noise = np.random.normal(0, 1, (height, width)).astype(np.float32)
        irregular_sigma = grain_size / (irregularity * 4 + 1)
        if irregular_sigma > 0:
            irregular_noise = ndimage.gaussian_filter(irregular_noise, irregular_sigma)
        
        # Blend the two noise patterns
        blend_factor = np.clip(irregularity, 0, 1)
        noise = (1 - blend_factor) * noise + blend_factor * irregular_noise
    
    # Normalize to range approximately -1 to 1
    noise = noise / (np.std(noise) * 3)
    noise = np.clip(noise, -1, 1)
    
    return noise


def generate_color_grain(height, width, grain_size, r_scale=1.0, g_scale=0.9, b_scale=0.8, 
                         color_correlation=0.6, irregularity=0.5, seed=None):
    """
    Generate partially correlated RGB grain
    
    Args:
        height (int): Image height
        width (int): Image width
        grain_size (float): Size of grain particles
        r_scale (float): Relative intensity for red channel
        g_scale (float): Relative intensity for green channel
        b_scale (float): Relative intensity for blue channel
        color_correlation (float): How correlated the channels are (0-1)
        irregularity (float): Irregularity of grain pattern (0-1)
        seed (int, optional): Random seed
        
    Returns:
        np.ndarray: RGB grain pattern as float32 array with range -1 to 1
    """
    if seed is not None:
        base_seed = seed
    else:
        base_seed = np.random.randint(0, 2**32 - 1)
    
    # Generate base noise pattern for partial correlation
    shared_noise = generate_base_noise(height, width, grain_size, irregularity, seed=base_seed)
    
    # Generate per-channel unique components
    r_unique = generate_base_noise(height, width, grain_size, irregularity, seed=base_seed+1)
    g_unique = generate_base_noise(height, width, grain_size, irregularity, seed=base_seed+2)
    b_unique = generate_base_noise(height, width, grain_size, irregularity, seed=base_seed+3)
    
    # Combine shared and unique components
    r_noise = color_correlation * shared_noise + (1 - color_correlation) * r_unique
    g_noise = color_correlation * shared_noise + (1 - color_correlation) * g_unique
    b_noise = color_correlation * shared_noise + (1 - color_correlation) * b_unique
    
    # Apply channel scaling
    r_noise *= r_scale
    g_noise *= g_scale
    b_noise *= b_scale
    
    # Stack to create RGB grain
    color_grain = np.stack([r_noise, g_noise, b_noise], axis=2)
    
    return color_grain


def modulate_by_luminance(grain, image, shadow_amount=1.5, midtone_amount=1.0, highlight_amount=0.5):
    """
    Modulate grain intensity based on image luminance
    
    Args:
        grain (np.ndarray): RGB grain pattern
        image (np.ndarray): Source image (0-1 range, float32)
        shadow_amount (float): Grain intensity in shadows
        midtone_amount (float): Grain intensity in midtones
        highlight_amount (float): Grain intensity in highlights
        
    Returns:
        np.ndarray: Luminance-modulated grain
    """
    # Calculate luminance (Rec. 709 coefficients)
    luminance = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
    
    # Create luminance-based modulation mask
    # This creates a curve where shadows get shadow_amount, midtones get midtone_amount,
    # and highlights get highlight_amount
    shadows_mask = 1.0 - np.clip(luminance * 2, 0, 1)
    highlights_mask = np.clip((luminance - 0.5) * 2, 0, 1)
    midtones_mask = 1.0 - shadows_mask - highlights_mask
    
    # Combine the masks with their respective amounts
    modulation = (shadow_amount * shadows_mask + 
                  midtone_amount * midtones_mask + 
                  highlight_amount * highlights_mask)
    
    # Apply modulation to each channel
    modulated_grain = grain * modulation[:,:,np.newaxis]
    
    return modulated_grain


def apply_grain(image, grain, intensity=1.0, blend_mode=BlendMode.OVERLAY):
    """
    Apply grain to image with specified blend mode
    
    Args:
        image (np.ndarray): Source image (0-1 range, float32)
        grain (np.ndarray): Grain pattern (-1 to 1 range, float32)
        intensity (float): Overall grain intensity multiplier
        blend_mode (BlendMode): Blend mode for grain application
        
    Returns:
        np.ndarray: Grained image
    """
    # Scale grain by intensity
    scaled_grain = grain * intensity
    
    if blend_mode == BlendMode.ADD:
        # Simple additive grain
        result = image + scaled_grain
        result = np.clip(result, 0, 1)
        
    elif blend_mode == BlendMode.OVERLAY:
        # Overlay blend mode (standard in Nuke)
        # For values < 0.5: 2 * base * blend
        # For values >= 0.5: 1 - 2 * (1 - base) * (1 - blend)
        bright_mask = image >= 0.5
        
        # Convert grain from -1...1 to 0...1 range for blending
        grain_01 = scaled_grain * 0.5 + 0.5
        
        result = np.zeros_like(image)
        result[~bright_mask] = 2 * image[~bright_mask] * grain_01[~bright_mask]
        result[bright_mask] = 1 - 2 * (1 - image[bright_mask]) * (1 - grain_01[bright_mask])
        result = np.clip(result, 0, 1)
        
    elif blend_mode == BlendMode.SOFT_LIGHT:
        # Soft Light blend (more subtle than overlay)
        # Convert grain from -1...1 to 0...1 range for blending
        grain_01 = scaled_grain * 0.5 + 0.5
        
        result = np.zeros_like(image)
        for c in range(3):
            result[:,:,c] = ((1 - 2 * grain_01[:,:,c]) * image[:,:,c]**2 + 
                            2 * grain_01[:,:,c] * image[:,:,c])
        result = np.clip(result, 0, 1)
    
    else:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")
    
    return result


def apply_film_grain(image, preset=GrainPreset.KODAK_5248, intensity_scale=1.0, 
                    blend_mode=BlendMode.OVERLAY, seed=None):
    """
    Apply film grain to an image using Nuke-style workflow
    
    Args:
        image (np.ndarray): Input image (0-1 range, float32, RGB)
        preset (GrainPreset): Film stock preset
        intensity_scale (float): Global scaling for grain intensity
        blend_mode (BlendMode): How grain is composited with the image
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Grained image
    """
    height, width = image.shape[:2]
    preset_values = preset.value
    
    # Generate color grain
    grain = generate_color_grain(
        height, width,
        grain_size=preset_values['size'],
        r_scale=preset_values['r_scale'],
        g_scale=preset_values['g_scale'],
        b_scale=preset_values['b_scale'],
        irregularity=preset_values['irregularity'],
        seed=seed
    )
    
    # Modulate grain by luminance (simulate film response)
    modulated_grain = modulate_by_luminance(
        grain, image,
        shadow_amount=1.5,  # More grain in shadows
        midtone_amount=1.0, # Normal grain in midtones
        highlight_amount=0.5  # Less grain in highlights
    )
    
    # Apply grain with specified intensity and blend mode
    final_intensity = preset_values['intensity'] * intensity_scale
    grained_image = apply_grain(
        image, 
        modulated_grain, 
        intensity=final_intensity,
        blend_mode=blend_mode
    )
    
    return grained_image


def process_image(input_path, output_path, preset=GrainPreset.KODAK_5248, 
                 intensity=1.0, blend_mode=BlendMode.OVERLAY, seed=None):
    """
    Process a single image file
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        preset (GrainPreset): Film stock preset
        intensity (float): Grain intensity multiplier
        blend_mode (BlendMode): Grain blend mode
        seed (int, optional): Random seed
    """
    # Read input image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to float32 and 0-1 range
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    
    # Convert BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply grain
    grained_img = apply_film_grain(
        img, 
        preset=preset, 
        intensity_scale=intensity,
        blend_mode=blend_mode,
        seed=seed
    )
    
    # Convert back to BGR for saving
    if len(grained_img.shape) == 3 and grained_img.shape[2] == 3:
        grained_img = cv2.cvtColor(grained_img, cv2.COLOR_RGB2BGR)
    
    # Convert to appropriate bit depth
    if input_path.lower().endswith(('.exr', '.hdr')):
        # Save as EXR/HDR
        cv2.imwrite(output_path, grained_img)
    elif input_path.lower().endswith('.png'):
        # Save as 16-bit PNG
        grained_img = np.clip(grained_img * 65535.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path, grained_img)
    else:
        # Save as 8-bit image
        grained_img = np.clip(grained_img * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, grained_img)


def process_sequence(input_dir, output_dir, file_pattern='*.exr', preset=GrainPreset.KODAK_5248, 
                    intensity=1.0, blend_mode=BlendMode.OVERLAY, 
                    temporal_variation=0.2, num_workers=None):
    """
    Process a sequence of images in parallel
    
    Args:
        input_dir (str): Directory containing input frames
        output_dir (str): Directory to save output frames
        file_pattern (str): Glob pattern to match input files
        preset (GrainPreset): Film stock preset
        intensity (float): Grain intensity multiplier
        blend_mode (BlendMode): Grain blend mode
        temporal_variation (float): How much grain varies between frames (0-1)
        num_workers (int, optional): Number of parallel workers (defaults to CPU count)
    """
    import glob
    
    # Set default number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input files
    input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    
    if not input_files:
        print(f"No files found matching pattern {file_pattern} in {input_dir}")
        return
    
    print(f"Processing {len(input_files)} frames...")
    
    # Generate base seed for the sequence
    base_seed = np.random.randint(0, 2**32 - 1)
    
    # Create processing function for parallel execution
    def process_frame(args):
        idx, input_file = args
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        # Calculate frame-specific seed for controlled temporal variation
        if temporal_variation > 0:
            # Mix base_seed with frame index for controlled variation
            frame_seed = base_seed + int(idx * temporal_variation * 1000)
        else:
            # Use same seed for all frames (static grain pattern)
            frame_seed = base_seed
        
        # Process the frame
        process_image(
            input_file, 
            output_file, 
            preset=preset,
            intensity=intensity,
            blend_mode=blend_mode,
            seed=frame_seed
        )
        
        return idx, filename
    
    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_frame, enumerate(input_files)))
    
    print(f"Completed processing {len(results)} frames")
    
def process_frame(args):
        input_file,output_file,preset,intensity,blend_mode = args
        
        # Process the frame
        process_image(
            input_file, 
            output_file, 
            preset=preset,
            intensity=intensity,
            blend_mode=blend_mode,
        )
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply film grain to images in Nuke style")
    parser.add_argument("--input", help="Input image path or directory")
    parser.add_argument("--output", help="Output image path or directory")
    parser.add_argument("--preset", choices=[p.name for p in GrainPreset], 
                       default="KODAK_5248", help="Film stock preset")
    parser.add_argument("--intensity", type=float, default=1.0, 
                       help="Grain intensity multiplier")
    parser.add_argument("--blend", choices=["ADD", "OVERLAY", "SOFT_LIGHT"], 
                       default="OVERLAY", help="Grain blend mode")
    parser.add_argument("--sequence", action="store_true", 
                       help="Process as image sequence")
    parser.add_argument("--pattern", default="*.png", 
                       help="File pattern for sequences (e.g. *.exr)")
    parser.add_argument("--temporal-variation", type=float, default=0, 
                       help="How much grain varies between frames (0-1)")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Set up parameters
    preset = GrainPreset[args.preset]
    blend_mode = BlendMode[args.blend]
    print(blend_mode)
    # Check if input is a directory (sequence) or single file
        # Process single image
    # process_image(
    #     args.input, 
    #     args.output,
    #     preset=preset,
    #     intensity=args.intensity,
    #     blend_mode=blend_mode
    # )
    img_pairs = []
    os.makedirs(args.output,exist_ok=True)
    for imgs in tqdm(os.listdir(args.input)):
        input_path = os.path.join(args.input,imgs)
        # denoise_path = os.path.join(args.dir3,imgs)
        # gt_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output,imgs)
        img_pairs.append((input_path,output_path,preset,args.intensity,blend_mode))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_frame,img_pairs)