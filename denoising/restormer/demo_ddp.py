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
from torch.utils.data import Dataset, DataLoader
import os
from lightning import Fabric
from lightning.fabric.utilities import rank_zero_only
import multiprocessing as mp
import time
mp.set_start_method('spawn', force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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


class ImageDataset(Dataset):
    def __init__(self, file_list, device, img_multiple_of=8):
        self.file_list = file_list
        self.device = device
        self.img_multiple_of = img_multiple_of

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_ = self.file_list[idx]
        img = cv2.cvtColor(cv2.imread(file_), cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(self.device)
        height, width = input_tensor.shape[2], input_tensor.shape[3]
        H = ((height + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((width + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - height if height % self.img_multiple_of != 0 else 0
        padw = W - width if width % self.img_multiple_of != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
        return input_tensor.squeeze(0), file_


def inference(input_tensor, model, args):
    """
    Runs inference on an input tensor using the provided model.
    Supports full-resolution or tile-wise inference with overlap.

    Parameters:
    -----------
    input_tensor : torch.Tensor
        Preprocessed image tensor of shape [1, C, H, W].
    model : torch.nn.Module
        Loaded model used for inference.
    args : argparse.Namespace
        Argument namespace with `.tile` and `.tile_overlap`.
    height : int
        Original image height.
    width : int
        Original image width.

    Returns:
    --------
    restored : np.ndarray
        Output image (uint8 RGB), post-processed from model output.
    """
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    
    _,_,height,width = input_tensor.shape
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


def ddp(fabric,args, files):
    fabric.seed_everything(42)
    model_params = {
        'inp_channels': 3, 'out_channels': 3, 'dim': 48,
        'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
        'bias': False, 'LayerNorm_type': 'BiasFree',
        'dual_pixel_task': False
    }
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_arch = run_path(args.model_arch)
    model = model_arch['Restormer'](**model_params)

    checkpoint = torch.load(args.weights_path)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    model = fabric.setup_module(model)
    dataset = ImageDataset(files, args.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    dataloader = fabric.setup_dataloaders(dataloader)

    with torch.no_grad():
        total_time = 0
        for input_tensor, path in tqdm(dataloader, desc="Processing"):
            input_tensor = fabric.to_device(input_tensor)
            torch.cuda.synchronize()
            start_time = time.time()
            output_img = inference(input_tensor, model, args)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            total_time += elapsed
            if rank_zero_only.rank == 0:
                save_img(os.path.join(result_dir, f"{os.path.splitext(os.path.basename(path[0]))[0]}.png"), output_img)
        
        if rank_zero_only.rank == 0:
            avg_time = total_time / len(dataloader)
            total_duration = time.strftime('%H:%M:%S', time.gmtime(total_time))
            print(f"Total inference time: {total_duration}")
            print(f"Average inference time per frame: {avg_time:.3f} s")


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

    if not files:
        raise Exception(f"No files found at {input_dir}")
    
    fabric = Fabric(accelerator="cuda", devices="auto", strategy="ddp")  # or specify devices=2
    fabric.launch(ddp, args, files)
    print(f"\nRestored images are saved at {result_dir}")