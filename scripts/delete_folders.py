import shutil
import os

# 1. Define the folders you want to move
folders_to_delete = [
    '/teamspace/lightning_storage/datasets/jhu-crowd-pp2'
]

for folder in folders_to_delete:
    # Always check if it exists first to avoid crashing
    if os.path.exists(folder):
        try:
            # rmtree removes the directory and ALL its contents
            shutil.rmtree(folder)
            print(f"Successfully deleted: {folder}")
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    else:
        print(f"Folder not found, skipping: {folder}")