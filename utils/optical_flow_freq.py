import os
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from util import adaptive_bilateral_blend,bilateral_frequency_blend

def load_frames(input_folder):
    """Load all images from a directory and return sorted list of frames."""
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = [cv2.imread(os.path.join(input_folder, f)) for f in image_files]
    return np.array(frames), image_files

def apply_gaussian_blur(frame, sigma=100):
    """Apply Gaussian blur to a frame."""
    return cv2.GaussianBlur(frame, (5, 5), sigma)

def bilateral_filter_luminance(image, d=7, sigma_spatial=15, sigma_range=8):
    """
    Applies bilateral filtering on the Y channel of an image in YCbCr space.

    Args:
        image (numpy.ndarray): Input RGB image.
        d (int): Diameter of the pixel neighborhood.
        sigma_spatial (float): Spatial Gaussian standard deviation (controls neighborhood size).
        sigma_range (float): Intensity Gaussian standard deviation (controls edge preservation).

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y,cb,cr = cv2.split(ycbcr)
    y_filtered = cv2.bilateralFilter(y, d, float(sigma_range), float(sigma_spatial))
    ycbcr_filtered = cv2.merge([y_filtered,cb,cr])
    denoised_rgb = cv2.cvtColor(ycbcr_filtered,cv2.COLOR_YCrCb2BGR)
    return denoised_rgb


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
    """Return neighboring frames with distance-based weighting."""
    num_frames = len(frames)

    # Select neighbor indices
    if index == 0:
        neighbor_indices = [0, 1, 2]
        temporal_distances = [0, 1, 2]
    elif index == 1:
        neighbor_indices = [1, 0, 2, 3]
        temporal_distances = [0, 1, 1, 2]
    elif index == num_frames - 1:
        neighbor_indices = [num_frames - 1, num_frames - 2, num_frames - 3]
        temporal_distances = [0, 1, 2]
    elif index == num_frames - 2:
        neighbor_indices = [num_frames - 2, num_frames - 3, num_frames - 1, num_frames - 4]
        temporal_distances = [0, 1, 1, 2]
    else:
        neighbor_indices = [index, index - 1, index + 1, index - 2, index + 2]
        temporal_distances = [0, 1, 1, 2, 2]

    # Extract neighbors
    neighbors = [frames[i] for i in neighbor_indices]
    
    # Compute optical flow magnitudes
    flow_magnitudes = np.array([np.linalg.norm(compute_optical_flow(frames[index], frames[i]), axis=-1) 
                               for i in neighbor_indices])
    
    # Create distance-based weights (closer frames get higher weight)
    distance_weights = np.array([1.0 if d == 0 else 
                                0.5 if d == 1 else 
                                0.1 if d == 2 else 
                                0.1 
                                for d in temporal_distances])
    
    # Apply distance weights to flow-based weights
    flow_weights = 1 / (1 + flow_magnitudes)  # Inverse weighting (H, W)
    for i in range(len(flow_weights)):
        flow_weights[i] = flow_weights[i] * distance_weights[i]
    
    # Normalize weights
    flow_weights /= np.sum(flow_weights, axis=0, keepdims=True)  
    
    return neighbors, flow_weights

def compute_adaptive_edge_mask(image, low_threshold=50, high_threshold=150, contrast_boost=1.5, dilation_iter=2):
    """Compute an enhanced adaptive edge mask with contrast boosting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Canny edges (binary edge map)
    edges = cv2.Canny(gray, low_threshold, high_threshold).astype(np.float32) / 255.0  

    # Compute Laplacian for contrast enhancement
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)  # Take absolute values to detect strong edges in both directions
    laplacian = np.clip(contrast_boost * laplacian, 0, 255) / 255.0  # Normalize

    # Combine Canny edges with contrast-boosted Laplacian edges
    adaptive_edge_map = np.maximum(edges, laplacian)  # Take max for adaptive enhancement

    # Dilate for smooth edge transitions
    kernel = np.ones((3, 3), np.uint8)
    adaptive_edge_map = cv2.dilate(adaptive_edge_map, kernel, iterations=dilation_iter)

    return adaptive_edge_map 

def blend_frames(ref_frame, warped_neighbors, weights, occlusion_mask,edge_mask=None):
    """
    Blend frames using per-pixel flow-based weights.
    
    Parameters:
      ref_frame: numpy array of shape (H, W, 3), the reference image.
      warped_neighbors: list of numpy arrays of shape (H, W, 3) for each neighbor.
      weights: numpy array of shape (N, H, W) containing per-pixel weights for each neighbor.
      occlusion_mask: numpy array of shape (H, W) with binary values (0: occluded, 1: valid).
    
    Returns:
      final_output: Blended image of shape (H, W, 3).
      
    Notes:
      - For pixels where occlusion_mask==0, the reference frame is used.
      - Weights are normalized per pixel across neighbors.
    """
    
    # Expand weights to (N, H, W, 1) for broadcasting across color channels.
    weights_expanded = weights[..., None]  # Now shape: (N, H, W, 1)
    if edge_mask is not None:
        weights = weights * (1 - edge_mask[None, :, :, 0]) 
    
    # Initialize blended_frame and per-pixel weight accumulator.
    blended_frame = np.zeros_like(ref_frame, dtype=np.float32)
    total_weight = np.zeros(ref_frame.shape[:2] + (1,), dtype=np.float32)  # Shape: (H, W, 1)
    
    # Loop over each neighbor and its per-pixel weight.
    for neighbor, w in zip(warped_neighbors, weights_expanded):
        neighbor = neighbor.astype(np.float32)
        blended_frame += neighbor * w
        total_weight += w

    # Prevent division by zero: clip per-pixel total weight.
    total_weight = np.clip(total_weight, 1e-6, None)
    blended_frame /= total_weight
    
    # Expand occlusion mask to (H, W, 1) for correct broadcasting.
    occlusion_mask_expanded = np.expand_dims(occlusion_mask, axis=-1)
    
    # For occluded regions (mask==0) use the reference frame.
    final_output = np.where(occlusion_mask_expanded == 1, blended_frame, ref_frame.astype(np.float32))
    
    return np.clip(final_output, 0, 255).astype(np.uint8)

def compute_pixel_difference_mask(ref_frame, neighbor_frame, sigma=1.0):
    """Compute per-pixel difference mask using YCrCb luminance and adaptive thresholding."""
    
    # Convert images to YCrCb color space
    ref_ycrcb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2YCrCb)
    neighbor_ycrcb = cv2.cvtColor(neighbor_frame, cv2.COLOR_BGR2YCrCb)

    # Extract luminance channel (Y)
    ref_y = ref_ycrcb[..., 0].astype(np.float32)
    neighbor_y = neighbor_ycrcb[..., 0].astype(np.float32)

    # Compute absolute luminance difference
    diff_luminance = np.abs(ref_y - neighbor_y)  # Shape: (H, W)
    
    # Compute adaptive threshold: mean + sigma * std deviation
    mean_diff = np.mean(diff_luminance)
    std_diff = np.std(diff_luminance)
    threshold = mean_diff + sigma * std_diff  # Adaptive threshold

    # Apply thresholding: areas with high difference get lower weight
    weighted_mask = np.exp(-diff_luminance / (threshold + 1e-6))  # Prevent div by zero

    return weighted_mask




def process_frames(input_folder, output_folder):
    """Main function: load frames, process, and save denoised frames."""
    frames, filenames = load_frames(input_folder)
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(frames)), desc="Processing Frames"):
        ref_frame = frames[i]
        neighbors, weights = get_neighboring_frames(frames, i)

        # Blur frames before computing optical flow
        ref_frame_blurred = bilateral_filter_luminance(ref_frame)
        neighbors_blurred = [bilateral_filter_luminance(frame) for frame in neighbors]
        
        edge_mask = compute_adaptive_edge_mask(ref_frame)
        edge_mask = np.expand_dims(edge_mask, axis=-1)
        

        
        cv2.imwrite(os.path.join('./tmp', filenames[i]), (edge_mask*255).astype('uint8'))

        warped_neighbors = []
        occlusion_masks = []
        pixel_diff_masks = []
        for original_frame, blurred_frame in zip(neighbors, neighbors_blurred):
            # Compute optical flow in both directions
            flow_fwd = compute_optical_flow(ref_frame_blurred, blurred_frame)
            flow_bwd = compute_optical_flow(blurred_frame, ref_frame_blurred)

            # Compute occlusion mask
            occlusion_mask = compute_occlusion_mask(flow_fwd, flow_bwd)
            pixel_diff_mask = compute_pixel_difference_mask(ref_frame_blurred, blurred_frame,sigma=2)

            # Warp the original sharp image using the computed flow
            warped_image = warp_image(original_frame, flow_fwd)

            warped_neighbors.append(warped_image)
            occlusion_masks.append(occlusion_mask)
            pixel_diff_masks.append(pixel_diff_mask)


        # Aggregate occlusion masks (if any neighbor is occluded, consider it occluded)
        final_occlusion_mask = np.max(np.stack(occlusion_masks, axis=0), axis=0)
        pixel_diff_masks = np.array(pixel_diff_masks)
        # pixel_diff_masks = np.expand_dims(pixel_diff_masks,axis=-1) # NxHxWx1

        weights = weights*pixel_diff_masks

        # Blend images while handling occlusions
        # denoised_frame = bilateral_frequency_blend(
        #     ref_frame, 
        #     warped_neighbors, 
        #     weights, 
        #     final_occlusion_mask,
        #     edge_mask
        # )
        denoised_frame = adaptive_bilateral_blend(
            ref_frame, 
            warped_neighbors, 
            weights, 
            final_occlusion_mask,
            edge_mask
        )

        # Save output
        cv2.imwrite(os.path.join(output_folder, filenames[i]), denoised_frame)

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
    process_frames(args.dir1, args.output_dir)
    # sptwo_denoise_folder(args.dir1,args.output_dir)