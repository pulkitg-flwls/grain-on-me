import cv2
import numpy as np
import matplotlib.pyplot as plt
from voronoi import generate_voronoi_pattern, scatter_voronoi_cells
from normalize import compute_response_curve,normalize_noise, add_adaptive_noise, \
    normalize_with_polynomial,fit_response_curve, tile_masked_region

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
        # Extract noise as difference - vectorized operation
        self.extracted_noise = noisy - denoised
        
        bin_centers, self.response_curve = compute_response_curve(self.extracted_noise, denoised_frame)
        self.fitted_curves, self.noise_func = fit_response_curve(bin_centers, self.response_curve)
        self.polynomials = [np.poly1d(p) for p in self.fitted_curves]

        return self.extracted_noise
    
    
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