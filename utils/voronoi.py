import numpy as np
import cv2
from scipy.spatial import Voronoi, distance_matrix
import matplotlib.pyplot as plt
# from noise import pnoise2

def replicate_grain_across_frame(grain_patch, frame_size):
    """Tiles the grain pattern to cover the entire frame."""
    rep_h = (frame_size[0] // grain_patch.shape[0]) + 1
    rep_w = (frame_size[1] // grain_patch.shape[1]) + 1
    tiled_grain = np.tile(grain_patch, (rep_h, rep_w, 1))
    return tiled_grain[:frame_size[0], :frame_size[1]]  # Crop to exact size

def generate_voronoi_pattern(noise_image, num_cells=200, border_extension=50):
    """Apply a Voronoi pattern to a noise image while preserving original noise values."""
    h, w,c = noise_image.shape
    
    # Step 1: Generate random Voronoi seeds
    points = np.random.rand(num_cells, 2) * [w, h]  # Scale to image size
    
    # # Step 2: Compute Voronoi diagram
    # vor = Voronoi(points)
    extra_points = [
        [-border_extension, -border_extension], [w + border_extension, -border_extension],  # Top-left, Top-right
        [-border_extension, h + border_extension], [w + border_extension, h + border_extension],  # Bottom-left, Bottom-right
        [w//2, -border_extension], [w//2, h + border_extension],  # Mid-top, Mid-bottom
        [-border_extension, h//2], [w + border_extension, h//2],  # Mid-left, Mid-right
    ]
    
    points = np.vstack([points, extra_points])  # Combine original and extra points
    vor = Voronoi(points)
    
    # Step 3: Create a Voronoi cell map (each pixel assigned to closest seed)
    cell_map = np.full((h, w), -1, dtype=np.int32)  # Fix: Use np.int32 for OpenCV compatibility
    # voronoi_overlay = cv2.cvtColor((noise_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    voronoi_overlay = (noise_image * 255).astype(np.uint8)

    for i, region in enumerate(vor.regions):
        if not region or -1 in region:  # Ignore infinite regions
            continue
        polygon = np.array([vor.vertices[v] for v in region], np.int32)
        cv2.fillPoly(cell_map, [polygon], int(i))  # Assign integer cell index
        cv2.polylines(voronoi_overlay, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)  # Draw boundary

    return cell_map, voronoi_overlay, vor,points

def generate_perlin_noise(h, w, scale=50):
    """Generate smooth displacement maps using Perlin noise."""
    perlin_dx = np.zeros((h, w), dtype=np.float32)
    perlin_dy = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            perlin_dx[y, x] = pnoise2(x / scale, y / scale)
            perlin_dy[y, x] = pnoise2(y / scale, x / scale)

    # Normalize the displacement field
    perlin_dx = (perlin_dx - perlin_dx.min()) / (perlin_dx.max() - perlin_dx.min()) * 10 - 5
    perlin_dy = (perlin_dy - perlin_dy.min()) / (perlin_dy.max() - perlin_dy.min()) * 10 - 5
    return perlin_dx, perlin_dy

def scatter_voronoi_cells(noise_image, cell_map, vor,points, max_shift=10, method="perlin"):
    """Randomly translates Voronoi cells while keeping them connected."""
    h, w, c = noise_image.shape
    
    if method == "rigid":
        # Generate a single random shift for the entire pattern
        dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)
        
        # Roll both noise and cell_map by the same amount
        translated_noise = np.roll(noise_image, shift=(dy, dx), axis=(0, 1))
        translated_cell_map = np.roll(cell_map, shift=(dy, dx), axis=(0, 1))

    elif method == "perlin":
        # Apply Perlin noise warping to introduce structured distortion
        perlin_dx, perlin_dy = generate_perlin_noise(h, w, scale=50)
        perlin_dx = (perlin_dx * max_shift).astype(int)
        perlin_dy = (perlin_dy * max_shift).astype(int)

        translated_noise = np.zeros_like(noise_image)
        translated_cell_map = np.zeros_like(cell_map)

        for y in range(h):
            for x in range(w):
                new_x = np.clip(x + perlin_dx[y, x], 0, w - 1)
                new_y = np.clip(y + perlin_dy[y, x], 0, h - 1)
                translated_noise[new_y, new_x] = noise_image[y, x]
                translated_cell_map[new_y, new_x] = cell_map[y, x]

    elif method == "voronoi_swap":
        # Swap entire Voronoi cells while keeping structure intact
        unique_cells = np.unique(cell_map)
        np.random.shuffle(unique_cells)

        translated_noise = np.zeros_like(noise_image)
        translated_cell_map = np.copy(cell_map)

        for cell_id in unique_cells:
            mask = (cell_map == cell_id)  # Get pixels of the current cell
            available_cells = unique_cells[unique_cells != cell_id]  # Avoid swapping with itself
            
            if available_cells.size == 0:  # Safety check
                continue
            
            target_cell = np.random.choice(available_cells)  # Select a random target cell
            target_mask = (cell_map == target_cell)

            # Flatten indices of both masks
            source_indices = np.where(mask)  # Get (y, x) positions of the source cell
            target_indices = np.where(target_mask)  # Get (y, x) positions of the target cell

            # Ensure source and target have enough pixels
            min_size = min(len(source_indices[0]), len(target_indices[0]))
            if min_size == 0:
                continue  # Skip if one of the cells has no pixels

            # Shuffle source pixels
            source_pixels = noise_image[source_indices]  # Get RGB values from the source cell
            np.random.shuffle(source_pixels)  # Shuffle only the color values

            # Apply shuffled pixels to the target locations
            translated_noise[target_indices[0][:min_size], target_indices[1][:min_size]] = source_pixels[:min_size]

    elif method == "local_shuffle":
        # Randomly shuffle pixels inside each Voronoi cell
        translated_noise = np.zeros_like(noise_image)
        translated_cell_map = np.copy(cell_map)

        for cell_id in np.unique(cell_map):
            if cell_id == -1:
                continue
            mask = (cell_map == cell_id)
            pixels = noise_image[mask]
            np.random.shuffle(pixels)  # Shuffle pixel order within cell
            translated_noise[mask] = pixels
    elif method == "voronoi_sample":
        dist_matrix = distance_matrix(points, points)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore zero distances (self-distance)
        # min_dist = np.min(dist_matrix)  # Find the minimum distance between any two centers
        min_dists = np.partition(dist_matrix, 3, axis=1)[:, 3]
        # Ensure shifts are at least min_dist but not exceeding max_shift
        # shift_range = (max(min_dists, max_shift), max_shift)  # Ensures minimum shift
        # Assign a valid shift for each Voronoi cell
        unique_cells = np.unique(cell_map)
        shift_dict = {}
        # shift_dict = {cell: np.random.randint(shift_range[0], shift_range[1]+1, size=2) * np.random.choice([-1, 1], size=2)
        #             for cell in unique_cells if cell != -1}
        # print(shift_dict)
        # unique_cells = np.unique(cell_map)
        # shift_dict = {cell: np.random.randint(-max_shift, max_shift+1, size=2) for cell in unique_cells if cell != -1}

        # Output image
        translated_noise = np.zeros_like(noise_image)
        for cell_idx, cell in enumerate(unique_cells):
            if cell == -1 or cell_idx >= len(min_dists):
                continue
            
            min_shift = int(min_dists[cell_idx])  # Ensure shift is at least this minimum distance
            min_shift = max(min_shift, 5)  # Ensure a reasonable lower bound
            max_shift = max(min_shift, max_shift)  # Prevent shift from being too low

            # Assign a random shift that is at least min_shift
            dx = np.random.randint(min_shift, max_shift + 1) * np.random.choice([-1, 1])
            dy = np.random.randint(min_shift, max_shift + 1) * np.random.choice([-1, 1])
            shift_dict[cell] = (dx, dy)
        # print(shift_dict)
        for cell_id in unique_cells:
            if cell_id == -1:
                continue  # Skip background
            
            mask = (cell_map == cell_id)
            dx, dy = shift_dict[cell_id]  # Get shift for this cell
            
            # Compute new pixel positions with wrapping
            y_indices, x_indices = np.where(mask)
            src_x = (x_indices - dx) % w  # Find source pixel (reverse shift)
            src_y = (y_indices - dy) % h

            # Assign the displaced pixels
            translated_noise[y_indices, x_indices] = noise_image[src_y, src_x]
    else:
        raise ValueError("Invalid method. Choose from ['rigid_translation', 'perlin', 'voronoi_swap', 'local_shuffle'].")

    # Create a new overlay for the translated Voronoi pattern
    translated_overlay = (translated_noise * 255).astype(np.uint8)

    # Draw translated Voronoi boundaries
    for region in vor.regions:
        if not region or -1 in region:  # Ignore infinite regions
            continue
        polygon = np.array([vor.vertices[v] for v in region], np.int32)
        cv2.polylines(translated_overlay, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)

    return translated_noise, translated_overlay

if __name__ == "__main__":

    # Generate noise image
    image_size = (256, 512,3)
    # noise_image = np.random.rand(*image_size)
    noise_image = cv2.imread('noise/extracted_noise.png',-1)/255.0
    # noise_image = cv2.resize(noise_image,(512,256),cv2.INTER_AREA)/255.0
    # noise_image = cv2.cvtColor(noise_image,cv2.COLOR_BGR2RGB)
    # Apply Voronoi pattern
    cell_map, voronoi_overlay, vor,points = generate_voronoi_pattern(noise_image)

    # Scatter the Voronoi cells
    scattered_noise,translated_overlay= scatter_voronoi_cells(noise_image, cell_map,vor,points, max_shift=20,method='voronoi_sample')



    # Save images
    # cv2.imwrite("original_noise.png", (noise_image * 255).astype(np.uint8))
    cv2.imwrite("voronoi_overlay.png", voronoi_overlay)
    # cv2.imwrite("scattered_noise.png", (scattered_noise * 255).astype(np.uint8))
    cv2.imwrite("translated_voronoi_overlay.png", translated_overlay)

    # plt.figure(figsize=(5,5))
    # plt.subplot(2,1,1)
    # plt.imshow(cv2.cvtColor(voronoi_overlay,cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis("off")
    

    # plt.subplot(2,1,2)
    # plt.imshow(cv2.cvtColor(translated_overlay,cv2.COLOR_BGR2RGB))
    # plt.title("Voronoi Sample")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig("voronoi_sample.png")