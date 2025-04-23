import numpy as np
import cv2
from scipy import ndimage
import os
from enum import Enum
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--dir3",type=str,default="")
    parser.add_argument("--title",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

class GrainPreset(Enum):
    """Film stock presets mimicking Nuke's options"""
    KODAK_5248 = {'size': 3.0, 'intensity': 0.4, 'irregularity': 0.6, 'r_scale': 1.0, 'g_scale': 0.7, 'b_scale': 0.9}
    KODAK_5279 = {'size': 2.5, 'intensity': 0.5, 'irregularity': 0.5, 'r_scale': 1.0, 'g_scale': 0.8, 'b_scale': 0.9}
    KODAK_VISION3 = {'size': 1.8, 'intensity': 0.3, 'irregularity': 0.4, 'r_scale': 1.0, 'g_scale': 0.9, 'b_scale': 0.8}
    FUJI_3510 = {'size': 2.0, 'intensity': 0.35, 'irregularity': 0.5, 'r_scale': 1.0, 'g_scale': 0.9, 'b_scale': 0.7}
    ILFORD_HP5 = {'size': 3.5, 'intensity': 0.6, 'irregularity': 0.7, 'r_scale': 1.0, 'g_scale': 1.0, 'b_scale': 1.0}


class BlendMode(Enum):
    """Blend modes for grain application"""
    ADD = 1
    OVERLAY = 2
    SOFT_LIGHT = 3


def generate_base_noise(height, width, grain_size):
    """Generate Perlin noise-based grain"""
    noise = np.random.normal(0, 1, (height, width)).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=grain_size)
    return (noise - noise.min()) / (noise.max() - noise.min())


def apply_grain(image, preset, shadow_intensity=0.5, midtone_intensity=0.3, highlight_intensity=0.2, blend_mode=BlendMode.OVERLAY):
    """Apply film grain with luminance-based variation"""
    height, width, _ = image.shape
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    
    # Generate noise for different luminance regions
    shadow_mask = np.clip(1 - grayscale * 2, 0, 1)
    highlight_mask = np.clip((grayscale - 0.5) * 2, 0, 1)
    midtone_mask = 1 - (shadow_mask + highlight_mask)
    
    base_noise = generate_base_noise(height, width, preset.value['size'])
    
    grain = (shadow_mask * shadow_intensity +
             midtone_mask * midtone_intensity +
             highlight_mask * highlight_intensity) * base_noise
    
    grain = np.stack([grain * preset.value['r_scale'],
                      grain * preset.value['g_scale'],
                      grain * preset.value['b_scale']], axis=-1)
    
    if blend_mode == BlendMode.ADD:
        result = np.clip(image / 255.0 + grain, 0, 1) * 255
    elif blend_mode == BlendMode.OVERLAY:
        result = np.clip(image / 255.0 * (1 - grain) + grain, 0, 1) * 255
    elif blend_mode == BlendMode.SOFT_LIGHT:
        result = np.clip((1 - (1 - image / 255.0) * (1 - grain)), 0, 1) * 255
    else:
        result = image
    
    return result.astype(np.uint8)


def process_frame(frame, preset=GrainPreset.KODAK_VISION3):
    return apply_grain(frame, preset)


def process_frames(input_folder, output_folder, preset=GrainPreset.KODAK_VISION3):
    """Applies film grain to all frames in a directory and saves output"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Skipping {frame_file}: unable to read.")
            continue

        processed_frame = process_frame(frame, preset)
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, processed_frame)
        print(f"Processed {frame_file} -> {output_path}")


if __name__ == "__main__":
    args = parse_args()
    process_frames(args.input_dir, args.output_dir)