import numpy as np
from skimage.util import view_as_windows
import argparse
import cv2
from normalize import compute_noise_map

def get_lowest_variance_patch_chunked(denoised_image, noisemap_image, patch_size, step=32, max_patches=100000):
    """
    Memory-efficient patch variance search for large RGB images.

    Args:
        denoised_image (np.ndarray): RGB denoised input image.
        noisemap_image (np.ndarray): RGB noisemap image.
        patch_size (tuple): (height, width) of patch.
        step (int): stride between patches.
        max_patches (int): chunk size for batch processing.

    Returns:
        Tuple[np.ndarray, np.ndarray, tuple]: Patch with lowest variance in denoised image,
                                              corresponding patch in noisemap,
                                              and top-left coordinate (y, x).
    """
    H, W, C = denoised_image.shape
    ph, pw = patch_size

    ys = np.arange(0, H - ph + 1, step)
    xs = np.arange(0, W - pw + 1, step)

    min_var = float('inf')
    min_denoised_patch = None
    min_noisemap_patch = None
    min_coord = (0, 0)

    coords = [(y, x) for y in ys for x in xs]

    for i in range(0, len(coords), max_patches):
        batch_coords = coords[i:i+max_patches]
        patches = np.array([
            denoised_image[y:y+ph, x:x+pw] for y, x in batch_coords
        ])  # shape: (N, ph, pw, C)

        patches_flat = patches.reshape(patches.shape[0], -1)
        vars = patches_flat.var(axis=1)

        min_idx = np.argmin(vars)
        if vars[min_idx] < min_var:
            min_var = vars[min_idx]
            y, x = batch_coords[min_idx]
            min_denoised_patch = denoised_image[y:y+ph, x:x+pw]
            min_noisemap_patch = noisemap_image[y:y+ph, x:x+pw]
            min_coord = (y, x)

    return min_denoised_patch, min_noisemap_patch, min_coord

def draw_patch_box(image, top_left, patch_size, color=(0, 255, 0), thickness=2):
    y, x = top_left
    h, w = patch_size
    image_with_box = image.copy()
    color = tuple(c / 255.0 for c in color)  # make color match image dtype
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), color, thickness)
    return image_with_box

def tile_noise_patch(patch, out_size):
    """
    Tiles the small noise patch to fill an output image of size out_size.

    Args:
        patch (np.ndarray): Input noise patch (H, W, C).
        out_size (tuple): Output image size (H, W).

    Returns:
        np.ndarray: Tiled noise image.
    """
    ph, pw, _ = patch.shape
    H, W = out_size
    rep_y = -(-H // ph)  # ceil division
    rep_x = -(-W // pw)
    tiled = np.tile(patch, (rep_y, rep_x, 1))
    return tiled[:H, :W]

def quilt_noise_patch(patch, out_size, patch_size, overlap):
    """
    Synthesizes a larger noise image via simple image quilting.

    Args:
        patch (np.ndarray): Input noise patch (H, W, C).
        out_size (tuple): Output image size (H, W).
        patch_size (int): Size of each patch to copy.
        overlap (int): Overlap between patches.

    Returns:
        np.ndarray: Quilted noise image.
    """
    H, W = out_size
    oh, ow, C = patch.shape
    step = patch_size - overlap
    ny = (H + step - 1) // step
    nx = (W + step - 1) // step

    canvas = np.zeros((ny * patch_size, nx * patch_size, C), dtype=patch.dtype)

    for j in range(ny):
        for i in range(nx):
            y = j * step
            x = i * step
            src_y = np.random.randint(0, oh - patch_size + 1)
            src_x = np.random.randint(0, ow - patch_size + 1)
            canvas[y:y+patch_size, x:x+patch_size] = patch[src_y:src_y+patch_size, src_x:src_x+patch_size]

    return canvas[:H, :W]

def fft_noise_sample(patch, out_size):
    """
    Synthesizes noise texture by replicating frequency domain characteristics.

    Args:
        patch (np.ndarray): Input noise patch (H, W, C).
        out_size (tuple): Output image size (H, W).

    Returns:
        np.ndarray: Synthesized noise image with similar frequency spectrum.
    """
    H, W = out_size
    out_image = np.zeros((H, W, patch.shape[2]), dtype=np.float32)

    for c in range(patch.shape[2]):
        f = np.fft.fft2(patch[:, :, c])
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        rand_phase = np.random.uniform(-np.pi, np.pi, size=(H, W))
        synth_fshift = mag.mean() * np.exp(1j * rand_phase)
        synth_f = np.fft.ifftshift(synth_fshift)
        out_image[:, :, c] = np.real(np.fft.ifft2(synth_f)).astype(np.float32)

    out_image -= out_image.min()
    out_image /= out_image.max() + 1e-8
    return out_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overwrite 3-channel images with their first channel.")
    parser.add_argument("--noisy_img", type=str)
    parser.add_argument("--denoised_img", type=str)
    parser.add_argument("--target_img", type=str)
    args = parser.parse_args()
    
    noisy_img = cv2.imread(args.noisy_img).astype(np.float32) / 255.0
    denoised_img = cv2.imread(args.denoised_img).astype(np.float32) / 255.0
    target_img = cv2.imread(args.target_img).astype(np.float32) / 255.0
    th,tw,tc = target_img.shape
    noise_map = compute_noise_map(denoised_img,noisy_img)
    patch = (128,128)
    small_denoised, small_noisy, min_coords = get_lowest_variance_patch_chunked(denoised_img,noise_map,patch_size=patch)
    visual = draw_patch_box(denoised_img, min_coords, patch)
    tile_noise = tile_noise_patch(small_noisy,(th,tw))
    quilt_noise = quilt_noise_patch(small_noisy,(th,tw),64,16)
    fft_noise = fft_noise_sample(small_noisy,(th,tw))
    cv2.imwrite('noise_transfer/small_denoised.png',(visual*255).astype('uint8'))
    cv2.imwrite('noise_transfer/small_noisy.png',(fft_noise*255).astype('uint8'))
    print(min_coords)