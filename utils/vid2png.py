import os
import cv2
import argparse
from pathlib import Path
import glob

def extract_frames(video_path, output_dir):
    video_name = Path(video_path).stem
    output_folder = os.path.join(output_dir, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"{frame_idx:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_path} to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument('--input_dir', required=True, help='Path to input folder containing videos')
    parser.add_argument('--output_dir', required=True, help='Path to output folder to save extracted frames')
    args = parser.parse_args()

    video_files = glob.glob(os.path.join(args.input_dir, 'Clip+-*.mp4'))
    if not video_files:
        print("No videos found with pattern Clip+-*.mp4")
        return

    for video_path in video_files:
        extract_frames(video_path, args.output_dir)

if __name__ == '__main__':
    main()
