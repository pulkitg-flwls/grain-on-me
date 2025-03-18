import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import skimage.metrics as metrics
from skimage.filters import laplace
import gc
from multiprocessing import Pool, cpu_count
import cv2
# import pyopenexr as exr

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--dir3",type=str,default="")
    parser.add_argument("--title",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

def crop_png(img,crop):
    x1,y1 = crop['top_left']
    x2,y2 = crop['bottom_right']
    img = img[y1:y2,x1:x2]
    return img

def read_png(img_path,):
    f = iio.imread(img_path)
    # f = cv2.imread(img_path)
    return f






def adaptive_contrast_enhancement(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
    
    :param image: Grayscale image
    :return: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def detect_face_region(image):
    """
    Detects the face region and returns a binary mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    mask = np.zeros_like(gray, dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 1
    return mask

def compute_adaptive_edge_difference(gt, pred, amplification=5.0):
    """
    Computes an enhanced, adaptive edge difference map with contrast boosting.
    """
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    # Enhance contrast in the ground truth to better detect lost details
    gt_gray = adaptive_contrast_enhancement(gt_gray)
    pred_gray = adaptive_contrast_enhancement(pred_gray)

    # Apply Canny edge detection (extracts high-frequency details)
    edges_gt = cv2.Canny(gt_gray, 50, 100)
    edges_pred = cv2.Canny(pred_gray, 50, 100)
    median_intensity = np.median(gt_gray)
    # canny_low = int(max(10, 0.33 * median_intensity))  # Adaptive lower bound
    # canny_high = int(min(255, 1.33 * median_intensity))  # Adaptive upper bound
    # edges_gt = cv2.Canny(gt_gray, canny_low, canny_high)
    # edges_pred = cv2.Canny(pred_gray, canny_low, canny_high)
    # Compute absolute edge difference
    edge_difference = np.abs(edges_gt - edges_pred)

    # Gaussian-based contrast normalization
    gaussian_weight = cv2.GaussianBlur(edge_difference.astype(np.float32), (7,7), 2.0)
    edge_difference = (edge_difference * gaussian_weight) ** 0.5  # Adaptive contrast boost

    # Apply face mask to suppress background noise
    face_mask = detect_face_region(gt)
    # edge_difference = edge_difference * face_mask

    # Amplify and normalize
    edge_difference *= amplification
    edge_difference = np.clip(edge_difference, 1e-5, 255)  # Avoid log(0)
    edge_difference = np.log1p(edge_difference)
    edge_difference = (edge_difference - edge_difference.min()) / (edge_difference.max() - edge_difference.min())

    return edge_difference

def plot_relative_absolute_error(grainy, denoise,crop,save_path,grainy_pics):
    """
    Computes and plots the Relative Absolute Error (RAE) between the given image and reference.

    RAE = np.abs(image - reference) / (reference + 0.0001)

    Args:
        image (numpy array): The denoised/degrained image.
        reference (numpy array): The clean ground-truth image.
    """
    # Ensure images are float32 for accurate computation
    denoise = denoise.astype(np.float32)
    grainy = grainy.astype(np.float32)
    
    denoise_crop = crop_png(denoise,crop)
    grainy_crop = crop_png(grainy,crop)
    
    grainy_pics_crop = {}
    for key in grainy_pics.keys():
        grainy_pics_crop[key] = crop_png(grainy_pics[key],crop)

    # rae_crop = crop_png(rae,crop)
    # rae_crop = crop_png(amplified_noise,crop)
    # rae_image = visualize_amplified_noise_overlay(grainy_crop,gt_crop)
    # rae_image = compute_adaptive_edge_difference(gt_crop.astype('uint8'), denoise_crop.astype('uint8'))
    # # Plot results
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    
    axes[0,0].imshow(denoise_crop.astype('uint8'), cmap='gray')
    axes[0,0].set_title("Denoised Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(grainy_pics_crop['tungsten_easy'].astype('uint8'), cmap='gray')
    axes[0,1].set_title("Easy: Tungsten")
    axes[0,1].axis("off")

    axes[0,2].imshow(grainy_pics_crop['arri_easy'].astype('uint8'), cmap='gray')
    axes[0,2].set_title("Easy: ARRI")
    axes[0,2].axis("off")

    axes[0,3].imshow(grainy_pics_crop['kodakvision3_easy'].astype('uint8'), cmap='gray')
    axes[0,3].set_title("Easy: Kodak Vision3")
    axes[0,3].axis("off")

    axes[1,0].imshow(grainy_crop.astype('uint8'), cmap='gray')
    axes[1,0].set_title("Grainy Image")
    axes[1,0].axis("off")

    axes[1,1].imshow(grainy_pics_crop['tungsten_hard'].astype('uint8'), cmap='gray')
    axes[1,1].set_title("Hard: Tungsten")
    axes[1,1].axis("off")

    axes[1,2].imshow(grainy_pics_crop['arri_hard'].astype('uint8'), cmap='gray')
    axes[1,2].set_title("Hard: ARRI")
    axes[1,2].axis("off")

    axes[1,3].imshow(grainy_pics_crop['kodakvision3_hard'].astype('uint8'), cmap='gray')
    axes[1,3].set_title("Hard: Kodak Vision3")
    axes[1,3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    del fig,axes,grainy,grainy_crop,denoise,denoise_crop,grainy_pics_crop
    gc.collect()


def process_single_image(args):
    denoised_path,grain_path,data_path,output_path = args
    denise_ufo={
        'top_left':(1540,278),
        'bottom_right':(2052,790)
    }
    rachel_vmd = {
        'top_left':(706,5),
        'bottom_right':(1218,517)
    }
    grainy = read_png(grain_path)
    denoise = read_png(denoised_path)
    

    grain_add = {
        'tungsten_easy','tungsten_hard','arri_easy', 'arri_hard','kodakvision3_easy','kodakvision3_hard'
    }
    grain_pics = {}
    filename = denoised_path.split('/')[-1]
    for key in grain_add:
        grain_pics[key] = read_png(os.path.join(data_path,key,'vmdf02/ep01_pt37_0140',filename))
    
    plot_relative_absolute_error(grainy,denoise,rachel_vmd,output_path,grain_pics)
    # plot_ycbcr(grainy,denoise,output_path)
    del grain_pics
    return 
    

if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'resize'),exist_ok=True)
    
    # nre, hfen, psnr,rae = [], [] ,[], []
    img_pairs = []
    for imgs in tqdm(os.listdir(args.dir1)):
        grainy_path = os.path.join(args.dir1,imgs)
        denoise_path = os.path.join(args.dir2,imgs)
        data_path = args.dir3
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((denoise_path,grainy_path,data_path,output_path))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)
    