import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
from pdb import set_trace as stx
import numpy as np


def parse_args():
    parser = ArgumentParser(description='Test Restormer on your own images')
    parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Directory of input images or path of single image')
    parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Directory for restored results')
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--model_arch', default='./basicsr/models/archs/restormer_arch.py', type=str, help='Model architecture')
    parser.add_argument('--weights_path', default='./pretrained_models/real_denoising.pth', type=str, help='Weights Path')
    args = parser.parse_args()
    return args

def load_img(filepath):
    """
    Loads an image from disk and converts it from BGR to RGB format.

    Parameters:
    -----------
    filepath : str
        Path to the input image file.

    Returns:
    --------
    img : np.ndarray
        Image in RGB format as a NumPy array.
    """
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    """
    Saves an image to disk after converting it from RGB to BGR format.

    Parameters:
    -----------
    filepath : str
        Path where the image will be saved.
    img : np.ndarray
        Image in RGB format as a NumPy array.
    """
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def prepare_input(file_,device,img_multiple_of=8):
    """
    Loads and preprocesses an image for inference:
    - Reads image
    - Normalizes pixel values to [0, 1]
    - Converts to CHW torch tensor
    - Pads to be divisible by `img_multiple_of`
    - Moves to specified device

    Parameters:
    -----------
    file_path : str
        Path to the input image.
    device : torch.device
        Device to move the tensor to (e.g., 'cuda' or 'cpu').
    img_multiple_of : int, optional
        Ensures both height and width are divisible by this value (default: 8).

    Returns:
    --------
    input_tensor : torch.Tensor
        Preprocessed and padded input tensor of shape [1, C, H, W].
    height : int
        Original image height.
    width : int
        Original image width.
    """
    img = load_img(file_)
    input_tensor = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
    # Pad the input if not_multiple_of 8
    height,width = input_tensor.shape[2], input_tensor.shape[3]
    H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-height if height%img_multiple_of!=0 else 0
    padw = W-width if width%img_multiple_of!=0 else 0
    input_tensor = F.pad(input_tensor, (0,padw,0,padh), 'reflect')
    return input_tensor,height,width


def inference(file_path, model, args):
    """
    Runs inference on an image using the provided model.
    Supports full-resolution or tile-wise inference with overlap.

    Parameters:
    -----------
    file_path : str
        Path to the input image.
    model : torch.nn.Module
        Loaded model used for inference.
    args : argparse.Namespace
        Argument namespace with at least `.tile` and `.tile_overlap` attributes.
    device : torch.device
        Device to run the inference on.

    Returns:
    --------
    restored : np.ndarray
        Output image (uint8 RGB), post-processed from model output.
    """
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    input_tensor, height, width = prepare_input(file_path,args.device)

    if args.tile is None:
        # Full image inference
        restored = model(input_tensor)
    else:
        # Tiled inference
        b, c, h, w = input_tensor.shape
        tile = min(args.tile, h, w)
        assert tile % 8 == 0, "Tile size must be a multiple of 8"
        tile_overlap = args.tile_overlap

        stride = tile - tile_overlap
        h_indices = list(range(0, h - tile, stride)) + [h - tile]
        w_indices = list(range(0, w - tile, stride)) + [w - tile]

        E = torch.zeros(b, c, h, w).type_as(input_tensor)
        W = torch.zeros_like(E)

        for h_idx in h_indices:
            for w_idx in w_indices:
                in_patch = input_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                E[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += out_patch
                W[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += 1

        restored = E / W

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :height, :width]
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    return img_as_ubyte(restored[0])

if __name__ == "__main__":
    args = parse_args()

    input_dir = args.input_dir
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    valid_exts = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    # Gather files
    if any(input_dir.endswith(ext) for ext in valid_exts):
        files = [input_dir]
    else:
        files = []
        for ext in valid_exts:
            files.extend(glob(os.path.join(input_dir, f'*.{ext}')))
        files = natsorted(files)

    if not files:
        raise Exception(f"No files found at {input_dir}")

    print(f"\nFound {len(files)} image(s) in {input_dir}")

    # Load model
    model_params = {
        'inp_channels': 3, 'out_channels': 3, 'dim': 48,
        'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
        'bias': False, 'LayerNorm_type': 'BiasFree',
        'dual_pixel_task': False
    }

    model_arch = run_path(args.model_arch)
    model = model_arch['Restormer'](**model_params)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)

    checkpoint = torch.load(args.weights_path)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    print(f"\n==> Running with weights: {args.weights_path}\n")

    # Run inference
    with torch.no_grad():
        for filepath in tqdm(files, desc="Processing"):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            output_img = inference(filepath,model,args)  # assuming `inference()` is defined
            save_img(os.path.join(result_dir, f"{filename}.png"), output_img)

    print(f"\nRestored images are saved at {result_dir}")