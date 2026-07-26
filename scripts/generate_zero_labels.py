import os
import glob
import numpy as np
from PIL import Image

def generate_negative_gt(image_dir, output_dir):
    """
    Generates ground truth .npy files for negative crowd counting samples.
    
    Parameters:
    - image_dir: Path to the directory containing the negative images.
    - output_dir: Path to save the generated .npy files.
    - label_format: 'density_map' (2D array of zeros) or 'points' (empty coordinate array).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all common image formats in the directory
    valid_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        # Also check for uppercase extensions (e.g., .JPG)
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in {image_dir}.")
        return

    print(f"Found {len(image_paths)} images. Generating .npy labels...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        
            # Create a 2D array of zeros matching the image's height and width
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Ground truth density maps are typically float32
        gt_array = np.zeros((height, width), dtype=np.float32)
            
        # Save the numpy array
        save_path = os.path.join(output_dir, f"{name}.npy")
        np.save(save_path, gt_array)
        
    print(f"Successfully generated {len(image_paths)} .npy files in '{output_dir}'.")

# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == "__main__":
    # Define your directories here
    INPUT_IMAGE_DIR = "/home/jl_fs/workspace/projects/crowd_counting/review_zeros/zeros_from_jhu/images"
    OUTPUT_NPY_DIR = "/home/jl_fs/workspace/projects/crowd_counting/review_zeros/zeros_from_jhu/ground-truth-npy"
    
    # Choose your format:
    # 'density_map' -> Creates a matrix of 0.0s the same size as the image.
    # 'points'      -> Creates an empty coordinate array [].    
    generate_negative_gt(INPUT_IMAGE_DIR, OUTPUT_NPY_DIR)