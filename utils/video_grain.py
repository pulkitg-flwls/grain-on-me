import cv2
import numpy as np
from renoise import apply_film_grain, GrainPreset, BlendMode
from tqdm import tqdm
import os
import gc
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--easy_int", type=float, default=0.1)
    parser.add_argument("--hard_int", type=float, default=0.4)
    parser.add_argument("--preset", choices=[p.name for p in GrainPreset], 
                       default="KODAK_5248", help="Film stock preset")
    parser.add_argument("--blend", choices=["ADD", "OVERLAY", "SOFT_LIGHT"], 
                       default="OVERLAY", help="Grain blend mode")
    args = parser.parse_args()
    return args

def process_video(input_path, output_path, preset=GrainPreset.KODAK_VISION3, intensity_scale=1.0,blend_mode=BlendMode.ADD):
    """
    Reads a video file, applies film grain to each frame, and saves the output.
    
    Args:
        input_path (str): Path to input video file.
        output_path (str): Path to save the processed video.
        preset (GrainPreset): Film stock preset for grain.
        intensity_scale (float): Scaling factor for grain intensity.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        with tqdm(total=frame_count, desc=f"Processing {os.path.basename(input_path)}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                
                # Convert frame to float32 (0-1 range) and apply grain
                frame = frame.astype(np.float32) / 255.0
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                grained_frame = apply_film_grain(frame, preset=preset, intensity_scale=intensity_scale, blend_mode=blend_mode)
                
                grained_frame = cv2.cvtColor(grained_frame, cv2.COLOR_RGB2BGR)
                grained_frame = (grained_frame * 255).astype(np.uint8)
                
                out.write(grained_frame)
                pbar.update(1)

                # Explicitly delete frame to free memory
                del frame, grained_frame
                gc.collect()  # Force garbage collection
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    
    finally:
        cap.release()
        out.release()
        gc.collect()
        print(f"Processed video saved at: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'easy'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'hard'), exist_ok=True)
    preset = GrainPreset[args.preset]
    blend_mode = BlendMode[args.blend]
    files = [p for p in os.listdir(args.input_dir) if p.endswith('.mp4')]
    
    for video in files[1:]:
        input_path = os.path.join(args.input_dir, video)
        easy_path = os.path.join(args.output_dir, 'easy', video)
        hard_path = os.path.join(args.output_dir, 'hard', video)

        # Process videos one by one to avoid OOM
        process_video(input_path, hard_path, preset=preset, intensity_scale=args.hard_int, blend_mode=blend_mode)
        process_video(input_path, easy_path, preset=preset, intensity_scale=args.easy_int, blend_mode=blend_mode)