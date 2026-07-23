import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

def convert_h5_to_npy(source_dir, dest_dir, dataset_key='density'):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all .h5 files
    h5_files = glob.glob(os.path.join(source_dir, '**', '*.h5'), recursive=True)
    
    if not h5_files:
        print(f"No .h5 files found in {source_dir}")
        return

    print(f"Found {len(h5_files)} files. Starting conversion...")

    for h5_path in tqdm(h5_files):
        try:
            # Recreate subdirectory structure if you have nested folders
            rel_path = os.path.relpath(h5_path, source_dir)
            npy_rel_path = os.path.splitext(rel_path)[0] + '.npy'
            npy_full_path = os.path.join(dest_dir, npy_rel_path)
            
            os.makedirs(os.path.dirname(npy_full_path), exist_ok=True)

            with h5py.File(h5_path, 'r') as f:
                if dataset_key in f:
                    data = f[dataset_key][:]
                    np.save(npy_full_path, data)
                else:
                    print(f"\nWarning: Key '{dataset_key}' missing in {h5_path}")
        except Exception as e:
            print(f"\nError on {h5_path}: {e}")

if __name__ == "__main__":
    # UPDATE THESE PATHS to match your actual dataset directories
    TRAIN_H5 = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/train/ground-truth-h5"
    TRAIN_NPY = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/train/ground-truth-npy"
    
    VAL_H5 = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/valid/ground-truth-h5"
    VAL_NPY = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/valid/ground-truth-npy"
    TEST_H5 = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/test/ground-truth-h5"
    TEST_NPY = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2/test/ground-truth-npy"
    
    print("Converting Training Set...")
    convert_h5_to_npy(TRAIN_H5, TRAIN_NPY)
    
    print("\nConverting Validation Set...")
    convert_h5_to_npy(VAL_H5, VAL_NPY)
    print("\nConverting Test Set...")
    convert_h5_to_npy(TEST_H5, TEST_NPY)