import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal, fftpack
from skimage.util import view_as_windows
from scipy.fftpack import fft2, fftshift
from voronoi import generate_voronoi_pattern, scatter_voronoi_cells
from normalize import compute_response_curve,normalize_noise, \
    add_adaptive_noise, fit_response_curve, tile_masked_region, normalize_with_polynomial

class ScatterGrain:
    """
    Class for applying and visualizing scatter grain for compositing.
    Optimized implementation using vectorized operations and library functions.
    """
    
    def __init__(self):
        """Initialize ScatterGrain processor"""
        self.noise_stats = None
        self.extracted_noise = None
    
    def png2exr(self,png):
        gamma=2.2
        png = (png/255.0).astype('float32')
        png = np.power(png,gamma)
        exr = png/(1-png)
        return exr
    def exr2png(self,exr):
        gamma=2.2
        image = np.clip(exr,0,None)
        image = image/(image+1)
        image = np.power(image,1/gamma)
        # image = (image*255).astype('uint8')
        return image
    
    def extract_noise(self, noisy_frame, denoised_frame):
        """
        Extract noise by subtracting denoised frame from noisy frame
        
        Parameters:
        -----------
        noisy_frame : numpy.ndarray
            Original frame with noise
        denoised_frame : numpy.ndarray
            Denoised version of the frame
        
        Returns:
        --------
        numpy.ndarray
            Extracted noise (difference between noisy and denoised)
        """
        # Ensure both frames are float32 for proper subtraction
        noisy = noisy_frame.astype(np.float32)
        denoised = denoised_frame.astype(np.float32)
        # noisy_lin = self.png2exr(noisy_frame)
        # denoised_lin = self.png2exr(denoised_frame)
        # Extract noise as difference - vectorized operation
        self.extracted_noise = noisy - denoised
        
        # self.extracted_noise = noisy_lin - denoised_lin
        
        # Calculate noise statistics
        self.noise_stats = self.analyze_noise_statistics(self.extracted_noise)
        bin_centers, self.response_curve = compute_response_curve(self.extracted_noise, denoised_frame)
        self.fitted_curves, self.noise_func = fit_response_curve(bin_centers, self.response_curve)
        self.polynomials = [np.poly1d(p) for p in self.fitted_curves]
        
        return self.extracted_noise
    
    def analyze_noise_statistics(self, noise, mask=None):
        """
        Analyze statistical properties of the extracted noise
        
        Parameters:
        -----------
        noise : numpy.ndarray
            Extracted noise pattern
        mask : numpy.ndarray, optional
            Optional mask to restrict analysis to specific regions
        
        Returns:
        --------
        dict
            Dictionary containing noise statistics
        """
        # If mask provided, apply it
        if mask is not None:
            if len(mask.shape) == 2 and len(noise.shape) == 3:
                mask = np.repeat(mask[:, :, np.newaxis], noise.shape[2], axis=2)
            
            # Use boolean indexing for masked analysis
            noise_masked = noise[mask > 0].reshape(-1, noise.shape[2])
        else:
            noise_masked = noise.reshape(-1, noise.shape[2])
        
        # Calculate basic statistics using numpy's optimized functions
        stats = {
            'mean': np.mean(noise_masked, axis=0),
            'std': np.std(noise_masked, axis=0),
            'min': np.min(noise_masked, axis=0),
            'max': np.max(noise_masked, axis=0)
        }
        
        # Convert to YCrCb for luminance/chrominance analysis
        if noise.shape[2] == 3:  # Only for RGB images
            # Add 0.5 offset to make the noise centered around 0.5 (for proper color conversion)
            noise_offset = noise + 0.5
            noise_ycrcb = cv2.cvtColor(noise_offset, cv2.COLOR_BGR2YCrCb)
            # Remove the offset after conversion
            noise_ycrcb = noise_ycrcb - 0.5
            
            if mask is not None:
                ycrcb_masked = noise_ycrcb[mask > 0].reshape(-1, noise_ycrcb.shape[2])
            else:
                ycrcb_masked = noise_ycrcb.reshape(-1, noise_ycrcb.shape[2])
            
            # Calculate YCrCb statistics
            stats['ycrcb_mean'] = np.mean(ycrcb_masked, axis=0)
            stats['ycrcb_std'] = np.std(ycrcb_masked, axis=0)
            
            # Use Y channel for frequency domain analysis
            y_channel = noise_ycrcb[:,:,0]
            
            # Calculate power spectrum using FFT
            f_transform = fftpack.fft2(y_channel)
            f_transform_shifted = fftpack.fftshift(f_transform)
            power_spectrum = np.abs(f_transform_shifted)**2
            
            # Store normalized power spectrum
            stats['power_spectrum'] = power_spectrum / np.max(power_spectrum)
        
        return stats
    
    def visualize_noise(self, noise=None, amplification=5.0, offset=0.5):
        """
        Visualize noise with amplification for better visibility
        
        Parameters:
        -----------
        noise : numpy.ndarray, optional
            Extracted noise pattern (if None, uses self.extracted_noise)
        amplification : float
            Factor to amplify noise (default: 5.0)
        offset : float
            Value to add to center the visualization (default: 0.5)
        
        Returns:
        --------
        numpy.ndarray
            Visualization of the noise pattern
        """
        if noise is None:
            if self.extracted_noise is None:
                raise ValueError("No noise available. Extract noise first or provide it as parameter.")
            noise = self.extracted_noise
        
        # Amplify noise and add offset to make it visible - vectorized operation
        vis_noise = noise * amplification + offset
        
        # Clip values to valid range [0, 1] - vectorized operation
        return np.clip(vis_noise, 0, 1)
    
    def generate_scattered_noise(self, shape, sample_ratio=0.1, seed=None):
        """
        Generate scattered noise by shuffling or re-sampling existing noise
        
        Parameters:
        -----------
        shape : tuple
            Shape of the desired noise array (height, width, channels)
        sample_ratio : float
            Proportion of pixels to sample from original noise (default: 0.1)
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        numpy.ndarray
            Scattered noise with shape matching the input shape
        """
        if self.extracted_noise is None:
            raise ValueError("No extracted noise available. Extract noise first.")
            
        if seed is not None:
            np.random.seed(seed)
        
        # Reshape noise for sampling - more efficient approach
        original_shape = self.extracted_noise.shape
        flat_noise = self.extracted_noise.reshape(-1, original_shape[2])
        
        # Determine number of samples and output size
        num_pixels = original_shape[0] * original_shape[1]
        num_samples = int(num_pixels * sample_ratio)
        out_pixels = shape[0] * shape[1]
        
        # Sample indices randomly
        sample_indices = np.random.choice(num_pixels, size=num_samples, replace=False)
        noise_samples = flat_noise[sample_indices]
        
        # Create random selection indices for each output pixel
        selection_indices = np.random.randint(0, num_samples, size=out_pixels)
        
        # Select random noise samples for each output pixel - vectorized approach
        scattered_flat = noise_samples[selection_indices]
        
        # Reshape to output dimensions
        scattered = scattered_flat.reshape(shape)
        
        return scattered
    
    def apply_random_transform(self,patch):
        """ Applies a random affine transformation (rotation, flipping, scaling) to a noise patch. """
        h, w,c = patch.shape
        center = (w // 2, h // 2)

        # Randomly choose rotation angle (0, 90, 180, 270 degrees)
        angle = np.random.choice([0, 90, 180, 270])

        # Randomly flip horizontally or vertically
        flip_h = np.random.choice([True, False])
        flip_v = np.random.choice([True, False])

        # Random rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_patch = cv2.warpAffine(patch, rotation_matrix, (w, h))

        # Apply flipping
        if flip_h:
            rotated_patch = cv2.flip(rotated_patch, 1)  # Horizontal flip
        if flip_v:
            rotated_patch = cv2.flip(rotated_patch, 0)  # Vertical flip

        return rotated_patch
    def compute_noise_statistics(self,noise, mask):
        """ Compute mean and standard deviation of noise in the non-edited region (per channel). """
        non_edited_pixels = noise[mask == 0]  # Extract only non-edited region
        mean_noise = np.mean(non_edited_pixels, axis=0)  # Mean per channel
        std_noise = np.std(non_edited_pixels, axis=0)  # Std dev per channel
        return mean_noise, std_noise

    def normalize_noise(self,scattered_noise, mean_target, std_target):
        """ Normalize scattered noise to match target mean and std dev per channel. """
        mean_input = np.mean(scattered_noise, axis=(0, 1))
        std_input = np.std(scattered_noise, axis=(0, 1))

        # Avoid division by zero
        std_input[std_input == 0] = 1

        # Normalize per channel
        normalized_noise = ((scattered_noise - mean_input) / std_input) * std_target + mean_target
        return np.clip(normalized_noise, 0, 255)  # Keep values in valid range


    def scatter_and_add_noise(self,noise, mask, min_patch_size=5, max_patch_size=20, alpha=0.2):
        """
        Scatter sampled noise patches from non-edited regions into the edited regions with affine transforms.

        Args:
            noise (numpy.ndarray): Extracted noise from the non-edited region (same shape as image).
            mask (numpy.ndarray): Binary mask indicating the edited region (1 for edited, 0 for non-edited).
            patch_size (int): Size of square patches to sample.
            num_patches (int): Number of patches to scatter.

        Returns:
            numpy.ndarray: Noise to be added to the edited regions.
        """
        edited_pixels = np.sum(mask)  # Total edited pixels

        if edited_pixels == 0:
            return np.zeros_like(noise)  # No edits, return zero noise

        # Step 2: Dynamically Set Patch Size
        grayscale_noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
        estimated_size = np.sqrt(edited_pixels) / 5
        noise_sharpness = cv2.Laplacian(grayscale_noise.astype('float64'), cv2.CV_64F).var()
        # patch_size = np.clip(int(np.sqrt(np.sum(mask)) / 5), min_patch_size, max_patch_size)
        # print(np.sqrt(np.sum(mask)) / 10)
        patch_size = 10
        
        # Step 3: Dynamically Set Number of Patches
        # num_patches = int(max(10, edited_pixels // 100))
        base_patches = edited_pixels / 100  # More patches for larger edits
        num_patches = int(np.clip(base_patches, 10, 500))
        # print("Num Patches",num_patches)
        # print("Patch Size",patch_size)
        mean_noise, std_noise = self.compute_noise_statistics(noise, mask)
        
        mask = (mask > 0).astype(np.uint8)
        non_edited_mask = 1 - mask
        non_edited_pixels = np.where(non_edited_mask > 0)

        # luminance = grayscale_noise/255
        # grain_strength = 1.0 - luminance  # More grain in dark areas, less in highlights
        # grain_strength = np.repeat(grain_strength[:, :, np.newaxis], 3, axis=2)
        # grain_strength *= mask

        if len(non_edited_pixels[0]) == 0:
            return np.zeros_like(noise)  # No non-edited regions to sample from

        
        patch_grid = view_as_windows(grayscale_noise, (patch_size, patch_size))
        h, w = patch_grid.shape[:2]
        patches = patch_grid.reshape(h * w, patch_size, patch_size)  # Flatten patches

        scattered_noise = np.zeros_like(noise)
        edited_pixels = np.where(mask > 0)

        if len(edited_pixels[0]) == 0:
            return np.zeros_like(noise)  # No edited regions to scatter noise

        for _ in range(num_patches):
            rand_idx = np.random.randint(len(non_edited_pixels[0]))
            
            y, x = non_edited_pixels[0][rand_idx], non_edited_pixels[1][rand_idx]

            y = max(0, min(y, h - 1))
            x = max(0, min(x, w - 1))
            noise_patch = patches[y * w + x]

            # Apply random affine transformation
            transformed_patch = self.apply_random_transform(noise_patch)
            transformed_patch_rgb = cv2.merge([transformed_patch] * 3)

            rand_idx_edit = np.random.randint(len(edited_pixels[0]))
            y_edit, x_edit = edited_pixels[0][rand_idx_edit], edited_pixels[1][rand_idx_edit]

            y_start = max(0, y_edit - patch_size // 2)
            x_start = max(0, x_edit - patch_size // 2)
            y_end = min(noise.shape[0], y_start + patch_size)
            x_end = min(noise.shape[1], x_start + patch_size)

            scattered_noise[y_start:y_end, x_start:x_end] += transformed_patch_rgb[:y_end - y_start, :x_end - x_start]
        # scattered_noise = self.normalize_noise(scattered_noise, mean_noise, std_noise)
        # scattered_noise = cv2.GaussianBlur(scattered_noise, (5, 5), 0)
        # return scattered_noise * mask
        return scattered_noise
        
    
    
    def generate_voronoi_grain(self,noise, mask, num_seeds=100, blend=True):
        """
        Applies Voronoi-based grain distribution using patches sampled from the noise.
        
        Args:
            noise (numpy.ndarray): Extracted noise (H, W, 3).
            mask (numpy.ndarray): Binary mask indicating the edited region (1 for edited, 0 for non-edited).
            num_seeds (int): Number of Voronoi seeds.
            blend (bool): Whether to apply Gaussian blur for blending.

        Returns:
            numpy.ndarray: The transformed noise distributed via Voronoi tessellation.
        """
        H, W, C = noise.shape

        normalized_noise = normalize_noise(noise)
        
        cell_map, voronoi_overlay, vor,points = generate_voronoi_pattern(normalized_noise,num_cells=500,border_extension=100)
        scattered_noise,translated_overlay= scatter_voronoi_cells(normalized_noise, cell_map,vor,points, max_shift=10,method='voronoi_sample')
        

        return scattered_noise
     
    
    def apply_matched_grain(self, original_plate, denoised_plate, edited_plate, edit_mask, blend_mode='add'):
        """
        Apply film grain from original plate to the edited plate, with special handling for edited regions
        
        Parameters:
        -----------
        original_plate : numpy.ndarray
            The original noisy footage (with natural film grain)
        denoised_plate : numpy.ndarray
            The denoised version of the original plate
        edited_plate : numpy.ndarray
            The edited footage after compositing/VFX work (denoised)
        edit_mask : numpy.ndarray
            Binary mask where 1 indicates edited regions
        blend_mode : str
            Blending mode to use ('add' or 'smooth_light')
        
        Returns:
        --------
        numpy.ndarray
            The final composited result with matching grain throughout
        """
        # 1. Extract the grain/noise from original footage
        self.extract_noise(original_plate, denoised_plate)
        # normalized_noise = normalize_noise(self.extracted_noise)
        normalized_noise = normalize_with_polynomial(self.extracted_noise,self.polynomials)
        # 2. Prepare the result image (start with edited plate)
        result = edited_plate.copy().astype('float32')
        # result = np.zeros(edited_plate.shape).astype('float32')
        # result_exr = self.png2exr(result)
       
        # 3. Ensure mask is in right format
        if len(edit_mask.shape) == 2:
            mask = np.repeat(edit_mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask = edit_mask.copy()
            
        mask = mask.astype(np.float32) / np.max(mask)
        inv_mask = 1.0 - mask
        
        # 4. Apply extracted noise to non-masked regions
        if blend_mode == 'add':
            # Simple additive blending - vectorized
            result = result + self.extracted_noise * inv_mask
            # result_exr = result + self.extracted_noise
        elif blend_mode == 'smooth_light':
            # Get masks for different noise values in non-edited regions
            neg_mask = (self.extracted_noise < 0) & (inv_mask > 0)
            pos_mask = (self.extracted_noise >= 0) & (inv_mask > 0)
            
            # Apply blending formula vectorized with where()
            result = np.where(
                neg_mask,
                result + 2 * self.extracted_noise * result,
                result
            )
            
            result = np.where(
                pos_mask,
                result + self.extracted_noise * (1 - result),
                result
            )
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}. Use 'add' or 'smooth_light'.")
        
        # 5. Generate and apply scattered noise to masked regions
        if np.any(mask > 0):
            
            # Generate scattered noise for edited regions
            # scattered_noise = self.generate_scattered_noise(edited_plate.shape)
            # scattered_noise = self.scatter_and_add_noise(self.extracted_noise,mask)
            tiled = tile_masked_region(normalized_noise,mask)
            scattered_noise = self.generate_voronoi_grain(normalized_noise,mask)
            # scattered_noise = self.generate_voronoi_grain(tiled,mask)
            adapted_noise = add_adaptive_noise(result, scattered_noise, self.fitted_curves,self.noise_func,mask)
            
            
            # Apply the scattered noise to masked regions
            if blend_mode == 'add':
                # Simple additive blending - vectorized
                result = result + adapted_noise *mask
                # result = result + self.extracted_noise*mask 
                # result,noise_stats = add_scattered_grain(self.extracted_noise,
                #                                          denoised_plate,result,mask)
                
            elif blend_mode == 'smooth_light':
                # Get masks for different noise values in edited regions
                neg_mask = (scattered_noise < 0) & (mask > 0)
                pos_mask = (scattered_noise >= 0) & (mask > 0)
                
                # Apply blending formula vectorized
                result = np.where(
                    neg_mask,
                    result + 2 * scattered_noise * result,
                    result
                )
                
                result = np.where(
                    pos_mask,
                    result + scattered_noise * (1 - result),
                    result
                )
        
        # 7. Clip values to valid range - vectorized
        # result = self.exr2png(result_exr)
        # result = result * 25
        return np.clip(result,0,1)
    
    def save_visualization(self, output_path, amplification=5.0):
        """
        Save visualization of the extracted noise
        
        Parameters:
        -----------
        output_path : str
            File path to save the visualization
        amplification : float
            Factor to amplify noise for visualization
        """
        if self.extracted_noise is None:
            raise ValueError("No extracted noise available. Extract noise first.")
        
        vis_noise = self.visualize_noise(amplification=amplification)
        
        # Convert to uint8 for saving - vectorized
        vis_noise_uint8 = (vis_noise * 255).astype(np.uint8)
        
        # Save visualization
        cv2.imwrite(output_path, vis_noise_uint8)
        print(f"Noise visualization saved to {output_path}")