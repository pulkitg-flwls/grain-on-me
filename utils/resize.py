import os
import imageio.v3 as iio
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import time
from skimage.transform import resize
import cv2
# import pyopenexr as exr

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def read_exr(filepath):
    """
    Reads an EXR file and returns a NumPy array (H, W, 3) in float32 format.
    Tries OpenCV first, falls back to OpenEXR.
    """
    try:
        # Attempt to load using OpenCV
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is not None and image.dtype == np.float32:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        else:
            print(f"[INFO] OpenCV failed to read EXR correctly. Using OpenEXR for {filepath}.")

    except Exception as e:
        print(f"[WARNING] OpenCV EXR reading error: {e}")

    # If OpenCV fails, use OpenEXR
    try:
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Read RGB channels as float32
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = exr_file.channels(["R", "G", "B"], pixel_type)

        # Convert to NumPy arrays
        r = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
        g = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)
        b = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)

        # Stack into RGB format (H, W, 3)
        image = np.stack([r, g, b], axis=-1)
        return image

    except Exception as e:
        print(f"[ERROR] Failed to load EXR: {e}")
        return None  # Return None if EXR could not be loaded

def write_exr(filepath, image):
    """
    Saves an EXR image from a NumPy array (H, W, 3) in float32 format.
    Tries OpenCV first, falls back to OpenEXR if needed.
    """
    try:
        # Ensure image is in float32 format
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Try saving with OpenCV
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"[INFO] EXR saved successfully with OpenCV: {filepath}")
        return True

    except Exception as e:
        print(f"[WARNING] OpenCV EXR saving failed: {e}")

    # If OpenCV fails, use OpenEXR
    try:
        header = OpenEXR.Header(image.shape[1], image.shape[0])
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Convert image to channels
        r = image[..., 0].astype(np.float32).tobytes()
        g = image[..., 1].astype(np.float32).tobytes()
        b = image[..., 2].astype(np.float32).tobytes()

        exr_file = OpenEXR.OutputFile(filepath, header)
        exr_file.writePixels({"R": r, "G": g, "B": b})

        print(f"[INFO] EXR saved successfully with OpenEXR: {filepath}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to save EXR: {e}")
        return False

def resize_exr(image):
    """
    Resizes an EXR image using anti-aliasing.
    - image: NumPy array (H, W, 3) in float32
    - new_size: tuple (new_height, new_width)
    Returns: Resized NumPy array (new_H, new_W, 3)
    """
    if image is None:
        print("[ERROR] Cannot resize a NoneType image.")
        return None
    h,w,c = image.shape
    resized_image = resize(image, (h//4,w//4), anti_aliasing=True, preserve_range=True).astype(np.float32)
    return resized_image
def exr_to_png(input_exr, output_png, bit_depth=8, gamma=2.2):
    """
    Converts an EXR file to PNG format with proper tone mapping.
    - input_exr: Path to the input EXR file.
    - output_png: Path to save the PNG file.
    - bit_depth: 8 (default) for standard PNG, 16 for higher precision.
    - gamma: Gamma correction value (default = 2.2 for sRGB-like conversion).
    """
    image = read_exr(input_exr)
    if image is None:
        print("[ERROR] Could not read EXR file.")
        return False

    # Normalize HDR to LDR (tone mapping)
    image = np.clip(image, 0, None)  # Ensure non-negative values
    image = image / (np.max(image) + 1e-6)  # Normalize to [0, 1]

    # Apply gamma correction
    image = np.power(image, 1.0 / gamma)

    # Convert to 8-bit or 16-bit format
    if bit_depth == 8:
        image = (image * 255).astype(np.uint8)
    elif bit_depth == 16:
        image = (image * 65535).astype(np.uint16)
    else:
        print("[ERROR] Unsupported bit depth. Use 8 or 16.")
        return False

    # Save PNG with OpenCV
    cv2.imwrite(output_png, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved PNG: {output_png} (Bit Depth: {bit_depth}-bit)")
    return True


if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    for imgs in os.listdir(args.dir1):
        # img = iio.imread(os.path.join(args.dir1, imgs))
        img = read_exr(os.path.join(args.dir1, imgs))
        resize_img = resize_exr(img)
        write_exr(os.path.join(args.output_dir,imgs),resize_img)
        # h,w,c = img.shape
        # img = (img-img.min())/(img.max()-img.min())
        exit()