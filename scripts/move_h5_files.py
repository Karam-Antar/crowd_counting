import os
import shutil

def move_h5_files(source_dir):
    # Define the destination folder path
    dest_dir = os.path.join(source_dir, "ground-truth-h5")
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    
    # Iterate through files in the source directory
    for filename in os.listdir('/teamspace/lightning_storage/datasets/jhu-crowd-pp2/valid'):
        # Check if the file is an .h5 file
        if filename.endswith(".h5"):
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            
            # Move the file
            shutil.move(source_file, dest_file)
            print(f"Moved: {filename} -> {dest_dir}")

# Specify the path to your folder
folders_paths = ["/teamspace/lightning_storage/datasets/jhu-crowd-pp2/val", "/teamspace/lightning_storage/datasets/jhu-crowd-pp2/train", "/teamspace/lightning_storage/datasets/jhu-crowd-pp2/test"]
move_h5_files("/teamspace/lightning_storage/datasets/jhu-crowd-pp2/val")
# for folder in folders_paths:
#     move_h5_files(folder)