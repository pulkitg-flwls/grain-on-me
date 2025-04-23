import os
import json
import cv2
import argparse

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return round(duration, 5)

def create_summary(base_dir):
    dataset_name = base_dir.split('/')[-4]
    # summary = {
    #     "dataset_name": f"dataset_name={os.path.basename(base_dir)}/dataset_version=v1"
    # }
    summary = {
        "dataset_name": f"{dataset_name}/dataset_version=v1"
    }
    for shot_name in os.listdir(base_dir):
        shot_id = shot_name.split('=')[-1]
        shot_path = os.path.join(base_dir, shot_name)
        test_masks_path = os.path.join(shot_path, "test/masks/face_mask")
        video_path = os.path.join(shot_path, "test/video", f"{shot_id}.mp4")

        if not os.path.isdir(test_masks_path) or not os.path.isfile(video_path):
            # print(os.path.isdir(test_masks_path))
            # print(video_path)
            continue

        num_frames = len([f for f in os.listdir(test_masks_path) if f.endswith(".png") or f.endswith(".jpg")])
        duration = get_video_duration(video_path)
        # if 'book' in shot_id:
        #     film_grain = True
        # elif 'vmd' in shot_id:
        #     film_grain = True
        # elif 'smug' or 'ufo' in shot_id:
        #     film_grain = False
        film_grain=True
        print(shot_id,film_grain)
        
        summary[shot_id] = {
            "simple_shot": True,
            "number_of_training_frames": num_frames,
            "number_of_test_frames": num_frames,
            "occlusion": False,
            "background_noise": False,
            "head_pose": "frontal",
            "heavy_film_grain": film_grain,
            "duration_t_train": duration,
            "duration_t_test": duration,
            "language": "en",
            "colorspace":"Output - Rec.709"
        }

    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ summary.json created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary.json for dataset")
    parser.add_argument("--base_dir", type=str, help="Path to the dataset base directory")
    args = parser.parse_args()

    create_summary(args.base_dir)