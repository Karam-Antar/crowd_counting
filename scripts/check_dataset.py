import os
from pathlib import Path

def verify_dataset_pairs(base_path):
    splits = ['train', 'valid', 'test']
    dataset_is_perfect = True

    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Error: Could not find the base directory '{base_path}'")
        return

    print(f"Checking dataset structure in: {base_path}\n" + "-"*40)

    for split in splits:
        print(f"Checking '{split}' split...")
        split_dir = base_dir / split
        
        img_dir = split_dir / 'images'
        gt_dir = split_dir / 'ground-truth-h5'
        
        if not img_dir.exists() or not gt_dir.exists():
            print(f"  [!] Missing 'images' or 'ground-truth-h5' directory in '{split}'")
            dataset_is_perfect = False
            continue

        # UPDATED LOGIC:
        # For images, we split the name at the first '_' and only keep the prefix (e.g., '0562')
        # This ignores the Roboflow hash so it can match the H5 file.
        img_stems = {f.stem.split('_')[0] for f in img_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        # H5 files are kept exactly as they are (e.g., '0562')
        gt_stems = {f.stem for f in gt_dir.iterdir() if f.is_file() and not f.name.startswith('.')}

        missing_gts = img_stems - gt_stems
        missing_imgs = gt_stems - img_stems

        if not missing_gts and not missing_imgs:
            print(f"  [✓] Perfect match! All {len(img_stems)} pairs are correct.")
        else:
            dataset_is_perfect = False
            if missing_gts:
                print(f"  [x] Error: {len(missing_gts)} images are missing their H5 ground truth files.")
                print(f"      Examples of missing GTs: {list(missing_gts)[:3]}")
            if missing_imgs:
                print(f"  [x] Error: {len(missing_imgs)} H5 files are missing their corresponding images.")
                print(f"      Examples of missing Images: {list(missing_imgs)[:3]}")
        print("-" * 40)

    if dataset_is_perfect:
        print("\nSUCCESS: Dataset is perfectly paired and ready for training!")
    else:
        print("\nWARNING: Dataset has mismatches. Please review the errors above.")

import os
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

def run_sanity_checks(base_path):
    splits = ['train', 'valid', 'test']
    base_dir = Path(base_path)
    
    corrupted_files = []
    suspicious_labels = []

    print("Starting automated sanity checks...")

    for split in splits:
        img_dir = base_dir / split / 'images'
        gt_dir = base_dir / split / 'ground-truth-h5'
        
        if not img_dir.exists(): continue

        for img_path in img_dir.iterdir():
            if not img_path.is_file() or img_path.name.startswith('.'):
                continue
                
            # 1. Image Corruption Check
            try:
                with Image.open(img_path) as img:
                    img.verify() # Quickly verifies the file is a valid image
            except Exception as e:
                corrupted_files.append((img_path.name, f"Image corruption: {e}"))
                continue

            # Reconstruct the corresponding H5 filename
            # (Using the split logic we established earlier)
            core_id = img_path.stem.split('_')[0]
            h5_path = gt_dir / f"{core_id}.h5"

            if not h5_path.exists():
                continue # We already checked for missing files in the previous script

            # 2. H5 Corruption & Value Check
            try:
                with h5py.File(h5_path, 'r') as hf:
                    # H5 files are dictionaries. We grab the first key to check the data.
                    # Usually, this is 'density' or 'points'
                    keys = list(hf.keys())
                    if not keys:
                        corrupted_files.append((h5_path.name, "H5 file is empty (no keys)."))
                        continue
                    
                    data = hf[keys[0]][:] # Load the actual array into memory
                    
                    # Check for NaNs or Infinities
                    if np.isnan(data).any() or np.isinf(data).any():
                        corrupted_files.append((h5_path.name, "Contains NaN or Inf values."))
                        continue
                        
                    # 3. Heuristic Check (Suspicious Counts)
                    # If it's a density map, the sum is the total crowd count.
                    # If it's an array of coordinates, the length is the crowd count.
                    estimated_count = np.sum(data) if len(data.shape) > 1 else len(data)
                    
                    if estimated_count == 0:
                        suspicious_labels.append((img_path.name, "Count is exactly 0."))
                    elif estimated_count > 10000: # Adjust this threshold as needed
                        suspicious_labels.append((img_path.name, f"Extremely high count: {estimated_count:.1f}"))
                        
            except Exception as e:
                corrupted_files.append((h5_path.name, f"H5 unreadable: {e}"))

    # Print Summary
    print("\n" + "="*40)
    print("SANITY CHECK RESULTS")
    print("="*40)
    
    if not corrupted_files and not suspicious_labels:
        print("All clear! No corrupted files or suspicious heuristics found.")
    else:
        print(f"Found {len(corrupted_files)} corrupted files.")
        for f, reason in corrupted_files[:10]:
            print(f"  - {f}: {reason}")
            
        print(f"\nFound {len(suspicious_labels)} suspicious labels (Requires manual review).")
        for f, reason in suspicious_labels[:10]:
            print(f"  - {f}: {reason}")

if __name__ == "__main__":
    from pathlib import Path

    # Replace this with the actual filename of the crowded image
    bad_image_name = '3360_jpg.rf.df37886bf850225d2e32e4103c33c8d1.jpg'

    dataset_path = Path('/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2')
    core_id = bad_image_name.split('_')[0]
    bad_h5_name = f'{core_id}.h5'

    deleted_count = 0

    for split in ['train', 'valid', 'test']:
        img_path = dataset_path / split / 'images' / bad_image_name
        h5_path = dataset_path / split / 'ground-truth-h5' / bad_h5_name
        
        if img_path.exists():
            img_path.unlink()
            print(f'Deleted image: {img_path}')
            deleted_count += 1
            
        if h5_path.exists():
            h5_path.unlink()
            print(f'Deleted label: {h5_path}')
            deleted_count += 1

    if deleted_count == 0:
        print('Could not find the files. Check the filename!')
    else:
        print('Bad data successfully purged!')


# if __name__ == "__main__":
#     DATASET_PATH = "/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2" 
#     verify_dataset_pairs(DATASET_PATH)


# import os
# import shutil
# from pathlib import Path

# dataset_path = Path('/home/jl_fs/workspace/projects/crowd_counting/datasets/jhu-crowd-pp-v2')
# review_dir = Path('/home/jl_fs/workspace/projects/crowd_counting/review_zeros')
# review_dir.mkdir(exist_ok=True)

# targets = [
#     '2030_jpg.rf.27ff3e90505d3b85a32d325e6c8c4ca2.jpg',
#     '1495_jpg.rf.d7e8ff111f02fc7d08b2ff76c1363828.jpg',
#     '3360_jpg.rf.df37886bf850225d2e32e4103c33c8d1.jpg',
#     '1564_jpg.rf.9d476c9295c41e3728a7e72d75440634.jpg',
#     '3139_jpg.rf.ade9d6799722481a373fd8b84a47797b.jpg',
#     '0157_jpg.rf.f6b41d4d1a425959afd770d675263bcb.jpg',
#     '3025_jpg.rf.6607baf7a1c29754c35871afc05e4c54.jpg',
#     '0104_jpg.rf.4eb2ebafd889bb145b8e880fe1ad7001.jpg',
#     '2227_jpg.rf.c5022f3317c2bd317de75276a101b9b6.jpg'
# ]

# for split in ['train', 'valid', 'test']:
#     img_dir = dataset_path / split / 'images'
#     if not img_dir.exists(): continue
#     for file in img_dir.iterdir():
#         if file.name in targets:
#             shutil.copy(file, review_dir / file.name)
#             print(f'Copied: {file.name}')