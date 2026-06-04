import os
import shutil
import numpy as np
import h5py

# 1. Define the mapping
splits_mapping = {
    "train": "train_densities",
    "valid": "val_densities",
    "test": "test_densities"
}

base_dir = "/teamspace/lightning_storage/datasets/jhu-crowd-pp-v2"

for split_name, density_folder in splits_mapping.items():
    print(f"--- Processing {split_name} split ---")
    
    split_dir = os.path.join(base_dir, split_name)
    # Define the new h5 directory path
    h5_dir = os.path.join(split_dir, "ground-truth-h5")
    labels_dir = os.path.join(split_dir, "labels")
    densities_dir = os.path.join(base_dir, density_folder)
    
    # 2. Create ground-truth-h5 folder and remove old labels folder
    if os.path.exists(labels_dir):
        print(f"Removing old labels folder: {labels_dir}")
        shutil.rmtree(labels_dir)
        
    os.makedirs(h5_dir, exist_ok=True)
        
    if not os.path.exists(densities_dir):
        print(f"Warning: Density folder {densities_dir} not found. Skipping...")
        continue
        
    # 3. Iterate, convert, and save into the new ground-truth-h5 folder
    npz_files = [f for f in os.listdir(densities_dir) if f.endswith('.npz')]
    
    for npz_filename in npz_files:
        npz_path = os.path.join(densities_dir, npz_filename)
        base_name = os.path.splitext(npz_filename)[0]
        h5_filename = f"{base_name}.h5"
        
        # Save path is now inside h5_dir
        h5_path = os.path.join(h5_dir, h5_filename)
        
        try:
            npz_data = np.load(npz_path)
            array_key = npz_data.files[0]
            density_array = npz_data[array_key]
            
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('density', data=density_array)
                
        except Exception as e:
            print(f"Error processing {npz_filename}: {e}")
            
    print(f"Successfully converted {len(npz_files)} files into {h5_dir}\n")

print("Conversion complete! The density files are now organized in 'ground-truth-h5' folders.")