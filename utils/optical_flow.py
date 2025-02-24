import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA



from argparse import ArgumentParser
def load_frames(input_folder):
    """Load all images from a directory and return sorted list of frames."""
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = [cv2.imread(os.path.join(input_folder, f)) for f in image_files]
    return np.array(frames), image_files

def apply_gaussian_blur(frame, sigma=100):
    """Apply Gaussian blur to a frame."""
    return cv2.GaussianBlur(frame, (5, 5), sigma)

def compute_optical_flow(ref_frame_blurred, target_frame_blurred):
    """Compute optical flow from target_frame to ref_frame using Farneback method."""
    ref_gray = cv2.cvtColor(ref_frame_blurred, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame_blurred, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(target_gray, ref_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = optical_flow.calc(target_gray, ref_gray, None)
    return flow  # Shape: (H, W, 2)
def compute_occlusion_mask(flow_fwd, flow_bwd, threshold=0.005):
    """Compute occlusion mask using forward-backward flow consistency.
    
    - Pixels where forward and backward flows disagree significantly are marked as occluded.
    """
    h, w = flow_fwd.shape[:2]

    # Compute inconsistency measure
    flow_sum = flow_fwd + cv2.remap(flow_bwd, flow_fwd[..., 0].astype(np.float32), 
                                    flow_fwd[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR)
    mask = np.linalg.norm(flow_sum, axis=2) > threshold  # Pixels where flow inconsistency is high
    
    return mask.astype(np.uint8)  # Binary mask (1 = occluded, 0 = valid)

def warp_image(original_frame, flow):
    """Warp an image using optical flow."""
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    new_x = np.clip(x + flow[..., 0], 0, w - 1)
    new_y = np.clip(y + flow[..., 1], 0, h - 1)

    warped = np.zeros_like(original_frame)
    for c in range(original_frame.shape[2]):
        warped[..., c] = cv2.remap(original_frame[..., c], new_x.astype(np.float32), new_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return warped

def get_neighboring_frames(frames, index):
    """Return neighboring frames and the original frame with boundary handling."""
    num_frames = len(frames)
    
    if index == 0:  # First frame
        neighbors = [frames[0], frames[1], frames[2]]  # Include original frame
        weights = np.array([0.75, 0.2, 0.05])  # Adjusted weights
    elif index == 1:  # Second frame
        neighbors = [frames[1], frames[0], frames[2], frames[3]]
        weights = np.array([0.65, 0.15, 0.15, 0.05])
    elif index == num_frames - 1:  # Last frame
        neighbors = [frames[num_frames - 1], frames[num_frames - 2], frames[num_frames - 3]]
        weights = np.array([0.75, 0.2, 0.05])
    elif index == num_frames - 2:  # Second last frame
        neighbors = [frames[num_frames - 2], frames[num_frames - 3], frames[num_frames - 1], frames[num_frames - 2]]
        weights = np.array([0.65, 0.15, 0.15, 0.05])
    else:
        neighbors = [frames[index], frames[index - 2], frames[index - 1], frames[index + 1], frames[index + 2]]
        weights = np.array([0.65, 0.05, 0.125, 0.125, 0.05])  # Original frame has the highest weight

    return neighbors, weights / np.sum(weights)  # Normalize weights

def blend_frames(ref_frame, warped_neighbors, weights, occlusion_mask):
    """Blend frames using weighted averaging, ensuring the original frame has the highest weight.
    
    - If occlusion_mask == 0 → Use the reference frame.
    - If occlusion_mask == 1 → Blend the warped frames.
    """
    blended_frame = np.zeros_like(ref_frame, dtype=np.float32)
    weight_sum = np.sum(weights)  # Normalize sum of weights

    for neighbor, weight in zip(warped_neighbors, weights):
        blended_frame += neighbor.astype(np.float32) * weight

    weight_sum = max(weight_sum, 1e-6)  # Prevent division by zero
    blended_frame /= weight_sum  # Normalize
    
    # Apply occlusion mask
    final_output = np.where(occlusion_mask[..., None] == 1, blended_frame, ref_frame)

    return np.clip(final_output, 0, 255).astype(np.uint8)

def extract_patches(image, patch_size=7):
    """Extract overlapping patches from an image and reshape correctly for PCA."""
    patches = view_as_windows(image, (patch_size, patch_size, 3))  # Extract patches
    
    # Reshape patches to (num_patches, patch_size*patch_size*channels)
    patches = patches.reshape(-1, patch_size, patch_size, 3)  
    return patches  # Shape: (N_patches, 7, 7, 3)

def pca_denoise(patches, variance_threshold=0.95):
    """Denoise patches using PCA."""
    n_patches, h, w, c = patches.shape  # Ensure correct unpacking
    reshaped_patches = patches.reshape(n_patches, -1)  # Flatten each patch into 1D
    
    pca = PCA()
    transformed_patches = pca.fit_transform(reshaped_patches)
    # Ensure at least 1 component is retained
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = max(1, np.searchsorted(explained_variance, variance_threshold) + 1)
    # num_components = min(num_components, transformed_patches.shape[1])
    # Adjust transformed_patches to match the required number of components
    transformed_patches_reduced = transformed_patches[:, :num_components]
    padded_transformed = np.zeros_like(transformed_patches)  # (5219352, 147)
    padded_transformed[:, :num_components] = transformed_patches_reduced  # Fill only first `num_components`

    # Perform PCA reconstruction
    # reconstructed = pca.inverse_transform(padded_transformed)
    reconstructed = pca.inverse_transform(transformed_patches)
    # Reshape back to original patch shape
    # print(reconstructed.max(),reconstructed.min())
    # return reconstructed.reshape(n_patches, h, w, c)
    return patches.reshape(n_patches, h, w, c)

def aggregate_patches(denoised_patches, weights, image_shape, patch_size=7, stride=1):
    """Reconstruct image from overlapping patches using weighted averaging."""
    H, W, C = image_shape  # Get original image size
    reconstructed_image = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W, C), dtype=np.float32)

    # Ensure weights are correctly reshaped to be broadcastable over patches
    weights = weights[:, None, None, None]  # Reshape to (N_patches, 1, 1, 1) for broadcasting

    # Calculate number of patches along each dimension
    num_patches_x = (W - patch_size) // stride + 1
    num_patches_y = (H - patch_size) // stride + 1

    patch_idx = 0  # Counter to track patch positions
    for y in range(0, num_patches_y * stride, stride):  # Iterate over height
        for x in range(0, num_patches_x * stride, stride):  # Iterate over width
            if patch_idx >= denoised_patches.shape[0]:  # Prevent out-of-bounds errors
                break

            patch = denoised_patches[patch_idx]  # Extract the denoised patch
            weight = weights[patch_idx]  # Extract corresponding weight (broadcastable)

            # Accumulate weighted sum for averaging
            reconstructed_image[y:y+patch_size, x:x+patch_size] += patch * weight
            weight_sum[y:y+patch_size, x:x+patch_size] += weight  # Store sum of weights

            patch_idx += 1  # Move to next patch

    # Normalize only where weight_sum > 0 to prevent NaN issues
    valid_mask = weight_sum > 0
    reconstructed_image[valid_mask] /= weight_sum[valid_mask]

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

def process_frames(input_folder, output_folder):
    """Main function: load frames, process, and save denoised frames."""
    frames, filenames = load_frames(input_folder)
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(frames)), desc="Processing Frames"):
        ref_frame = frames[i]
        neighbors, weights = get_neighboring_frames(frames, i)

        # Blur frames before computing optical flow
        ref_frame_blurred = apply_gaussian_blur(ref_frame)
        neighbors_blurred = [apply_gaussian_blur(frame) for frame in neighbors]

        warped_neighbors = []
        occlusion_masks = []

        for original_frame, blurred_frame in zip(neighbors, neighbors_blurred):
            # Compute optical flow in both directions
            flow_fwd = compute_optical_flow(ref_frame_blurred, blurred_frame)
            flow_bwd = compute_optical_flow(blurred_frame, ref_frame_blurred)

            # Compute occlusion mask
            occlusion_mask = compute_occlusion_mask(flow_fwd, flow_bwd)

            # Warp the original sharp image using the computed flow
            warped_image = warp_image(original_frame, flow_fwd)

            warped_neighbors.append(warped_image)
            occlusion_masks.append(occlusion_mask)

        # Aggregate occlusion masks (if any neighbor is occluded, consider it occluded)
        final_occlusion_mask = np.max(np.stack(occlusion_masks, axis=0), axis=0)

        # Blend images while handling occlusions
        denoised_frame = blend_frames(ref_frame_blurred, warped_neighbors, weights, final_occlusion_mask)

        # Apply median filter to handle edges
        denoised_frame = median_filter(denoised_frame, size=(3, 3, 1))

        # Save output
        cv2.imwrite(os.path.join(output_folder, filenames[i]), denoised_frame)

    print(f"Denoised frames saved to {output_folder}")

def sptwo_denoise_folder(input_folder, output_folder, patch_size=7):
    """Apply SPTWO denoising to a folder of frames."""
    frames, filenames = load_frames(input_folder)
    num_frames, h, w, c = frames.shape
    denoised_video = np.zeros_like(frames)

    for t in tqdm(range(1, num_frames - 1), desc="Denoising Frames"):
        prev_frame, curr_frame, next_frame = frames[t-1], frames[t], frames[t+1]

        # Compute optical flow
        flow_fwd = compute_optical_flow(curr_frame, next_frame)
        flow_bwd = compute_optical_flow(curr_frame, prev_frame)

        # Warp frames based on optical flow
        prev_warped = warp_image(prev_frame, flow_bwd)
        next_warped = warp_image(next_frame, flow_fwd)

        # Extract patches
        patches_curr = extract_patches(curr_frame, patch_size)
        patches_prev = extract_patches(prev_warped, patch_size)
        patches_next = extract_patches(next_warped, patch_size)

        # Stack patches and apply PCA denoising
        stacked_patches = np.concatenate([patches_prev, patches_curr, patches_next], axis=0)
        denoised_patches = pca_denoise(stacked_patches)
        num_neighbors = stacked_patches.shape[0] // patches_curr.shape[0]
        patches_curr_repeated = np.repeat(patches_curr, num_neighbors, axis=0)

        # Flatten the patches along spatial and channel dimensions
        stacked_patches_flat = stacked_patches.reshape(stacked_patches.shape[0], -1)  # (N, 147)
        patches_curr_flat = patches_curr_repeated.reshape(patches_curr_repeated.shape[0], -1)  # (N, 147)
        
        # Compute the norm correctly (Euclidean distance per patch)
        weights = np.exp(-np.linalg.norm(stacked_patches_flat - patches_curr_flat, axis=1))

# Ensure weights are correctly reshaped for aggregation
        weights = weights[:, None, None, None]  # Make weights broadcastable

        # Compute adaptive weights for aggregation
        # weights = np.exp(-np.linalg.norm(stacked_patches - patches_curr, axis=(-1, -2, -3)))  # Adaptive weights
        denoised_video[t] = aggregate_patches(denoised_patches, weights, image_shape=curr_frame.shape, patch_size=7)
        print(denoised_video[t].max(),denoised_video[t].min())
        cv2.imwrite(os.path.join(output_folder, filenames[t]), denoised_video[t])
    # Copy the first and last frame unmodified (since they can't be denoised properly)
    denoised_video[0] = frames[0]
    cv2.imwrite(os.path.join(output_folder, filenames[0]), denoised_video[0])
    denoised_video[-1] = frames[-1]
    cv2.imwrite(os.path.join(output_folder, filenames[-1]), denoised_video[-1])

    # Save denoised frames
    # os.makedirs(output_folder, exist_ok=True)
    # for i, filename in enumerate(filenames):
    #     cv2.imwrite(os.path.join(output_folder, filename), denoised_video[i])

    print(f"Denoised frames saved to {output_folder}")


def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # process_frames(args.dir1, args.output_dir)
    sptwo_denoise_folder(args.dir1,args.output_dir)