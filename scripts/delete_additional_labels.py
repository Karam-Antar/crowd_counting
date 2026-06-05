import os

# Set this to False only when you are 100% ready to delete the files
DRY_RUN = True

img_dir = "/teamspace/lightning_storage/datasets/jhu-crowd-pp-v2/valid/images"
h5_dir = "/teamspace/lightning_storage/datasets/jhu-crowd-pp-v2/valid/ground-truth-h5"

# Get sets of filenames without extensions
# Note: This assumes exact filename matches (e.g., '123.jpg' matches '123.h5')
imgs = {os.path.splitext(f)[0].split('_')[0] for f in os.listdir(img_dir)}
h5s = {os.path.splitext(f)[0] for f in os.listdir(h5_dir)}
print(f"Found {len(imgs)} images and {len(h5s)} label files.")

# Find files that are in one but not the other
missing_h5 = imgs - h5s
missing_imgs = h5s - imgs

print(f"--- Found {len(missing_imgs)} label files without a matching image ---")

# Iterate and delete the orphans
for f in missing_imgs:
    file_path = os.path.join(h5_dir, f + '.h5')
    
    if DRY_RUN:
        print(f"[DRY RUN] Would delete: {file_path}")
    else:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if DRY_RUN:
    print("\nDRY RUN mode is ON. No files were deleted. Change DRY_RUN to False to proceed.")
else:
    print("\nCleanup complete.")