import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
# from retinaface import RetinaFace
from argparse import ArgumentParser
from tqdm import tqdm
import json

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    parser.add_argument('--crop_json', type=str, help='JSON file with crop coordinates (x,y,width,height)')
    args = parser.parse_args()
    return args


def detect_face_with_mediapipe(image):
    """
    Detect face using MediaPipe Face Detection.
    """
    mp_face_detection = mp.solutions.face_detection
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range detection
        min_detection_confidence=0.5
    ) as face_detection:
        # Convert to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        ih, iw, _ = image.shape

        # Get the first detected face
        if len(results.detections) > 1:
            print(f"\nFound {len(results.detections)} faces in the first frame. Automatically selecting the largest face.")
            
            largest_face = None
            largest_area = 0
            
            # Find the largest face
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                
                area = w * h
                
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)
            
            # print(f"Selected the largest face with area: {largest_area} pixels")
            return largest_face
        else:
            # Get the only detected face
            detection = results.detections[0]
            
            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            
            return (x, y, w, h)

def detect_face_with_retinaface(image):
    """
    Detect face using RetinaFace detector.
    """
    # Detect faces
    faces = RetinaFace.detect_faces(image)
    
    if not faces or isinstance(faces, tuple):  # RetinaFace returns tuple when no faces are found
        return None
    
    # Get the first face
    face_key = list(faces.keys())[0]
    face = faces[face_key]
    
    # Get bounding box coordinates
    bbox = face["facial_area"]
    x1, y1, x2, y2 = bbox
    
    # Convert to x, y, width, height format
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    
    return (x, y, w, h)

def detect_face_in_first_frame(frames_dir, detector_type="mediapipe"):
    """
    Detect the face in the first frame using the specified detector
    and return coordinates for a 512x512 crop around the face.
    
    Parameters:
    - frames_dir: Directory containing image frames
    - detector_type: Type of detector to use ("mediapipe" or "retinaface")
    
    Returns:
    - Tuple of (x, y, width, height) for the crop
    """
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(frames_dir).glob(f'*{ext}')))
    
    # Sort the files to ensure we get the first frame
    image_files.sort()
    
    if not image_files:
        raise ValueError(f"No image files found in {frames_dir}")
    
    num_frames = len(image_files)
    # Load the first frame
    first_frame = cv2.imread(str(image_files[int(num_frames/2)]))
    
    if first_frame is None:
        raise ValueError(f"Could not read image file: {image_files[0]}")
    
    # Detect face using the specified detector
    print(f"Using {detector_type} face detector...")
    
    if detector_type.lower() == "mediapipe":
        face_coords = detect_face_with_mediapipe(first_frame)
    elif detector_type.lower() == "retinaface":
        face_coords = detect_face_with_retinaface(first_frame)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    if face_coords is None:
        print("No faces detected in the first frame. Using center of the image instead.")
        h, w = first_frame.shape[:2]
        # Default to center of the image
        return (w//2 - 256, h//2 - 256, 512, 512)
    
    # Extract face coordinates
    x, y, w, h = face_coords
    
    # Calculate the center of the face
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate a 512x512 bounding box around the face center
    x1 = max(0, center_x - 256)
    y1 = max(0, center_y - 256)
    
    print(f"Face detected at ({x}, {y}, {w}, {h})")
    print(f"Creating 512x512 crop at ({x1}, {y1})")
    
    return (x1, y1, 512, 512)

def crop_frames(frames_dir, output_dir, crop_coords):
    """
    Crop all frames using the provided coordinates and save to output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(frames_dir).glob(f'*{ext}')))
    
    # Sort the files to process them in order
    image_files.sort()
    
    # Extract crop coordinates
    x, y, crop_width, crop_height = crop_coords
    
    # Process each frame
    for img_path in tqdm(image_files):
        # Read the image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Could not read image file: {img_path}")
            continue
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Ensure crop stays within image boundaries
        end_x = min(x + crop_width, width)
        end_y = min(y + crop_height, height)
        
        # Adjust crop size if it goes beyond image boundaries
        actual_width = end_x - x
        actual_height = end_y - y
        
        if actual_width != crop_width or actual_height != crop_height:
            print(f"Warning: Crop size adjusted to {actual_width}x{actual_height} for {img_path}")
        
        # Crop the image
        cropped_img = img[y:end_y, x:end_x]
        
        # Save the cropped image
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), cropped_img)
        # print(f"Cropped {img_path.name} saved to {output_path}")

def main():
    # Set your input and output directories
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    
    detector_type = "mediapipe"
    
    try:
        # Detect face in the first frame
        # crop_coords = detect_face_in_first_frame(args.dir1, detector_type)
        # print(f"Using crop coordinates: {crop_coords}")
        if args.crop_json and os.path.exists(args.crop_json):
            print(f"Loading crop coordinates from {args.crop_json}")
            with open(args.crop_json, 'r') as f:
                crop_coords = tuple(json.load(f))
            print(f"Loaded crop coordinates: {crop_coords}")
        else:
            # Detect face in the first frame
            crop_coords = detect_face_in_first_frame(args.dir1, detector_type)
            print(f"Using crop coordinates: {crop_coords}")
            json_filename = os.path.join("./crop_coordinates.json")
            with open(json_filename, 'w') as f:
                json.dump(crop_coords, f)
            print(f"Saved crop coordinates to {json_filename}")
        # Crop all frames using the same coordinates
        if args.crop_json:
            crop_frames(args.dir1, args.output_dir, crop_coords)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()