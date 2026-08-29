## 1. Supported Datasets & Benchmarks

The system is engineered and evaluated using benchmark datasets specifically designed for dense crowd counting and severe perspective distortions:

| Dataset | Split (Train / Test) | Target Density Type | Primary Purpose |
| --- | --- | --- | --- |
| **ShanghaiTech Part A** | 300 / 182 | Adaptive Gaussian (`.npy`) | High-density internet-scraped crowds with extreme scale variations.
| **JHU-CROWD++** | 2,772 / 1,600 | Geometry-Adaptive (`.npy`) | Large-scale diverse scenes covering various weather conditions and lighting.

NOTE: in later phases of the project, 36 negative samples were added to ShanghaiTech Part A dataset to improve the diversity, and 2 or 3 mislabled samples were removed from JHU-CROWD++

---
## 2. Dataset Acquisition & Conversion Workflow

Official crowd counting releases typically provide raw point annotations (`.mat` or JSON point coordinates) rather than pre-generated density maps, so we downloaded the preprocessed version of each dataset from kaggle.

1. ShanghaiTech

```text 
kaggle datasets download tthien/shanghaitech-with-people-density-map
```

2. JHU-CROWD++

* to download the images:
```text
kaggle datasets download hoangxuanviet/jhu-crowd
```

* to download the labels in `.npz` format:
```text
kaggle datasets download swapnilbagde000/jhu-crowdv2-density-maps
```

then the `.npz` files should be converted into `.npy` format.
after that a mismatch will appear between the images and labels due to addition labels or additional images that does not exist on the other side, this mismatch is coming from kaggle data.

---

## 3. Local Disk Structure

To ensure the `CrowdDataModule` and custom `Dataset` classes can map images to their corresponding ground-truth targets without path or index mismatches, data must adhere to the following directory hierarchy:

```text
data/
└── ShanghaiTech/
    └── part_A/
        ├── train_data/
        │   ├── images/          # IMG_1.jpg, IMG_2.jpg...
        │   └── ground-truth-npy/ # IMG_1.npy, IMG_2.npy...
        └── test_data/
            ├── images/
            └── ground-truth-npy/

```

NOTE: the code relies on the order of the (img, density_map) pairs rather than the names, so the first image in the images folder will have the first .npy file in the ground-truth-npy folder as its density map label.
NOTE: for JHU-CROWD++ we can name the folder for validation data "valid" and it must have the same structure (images folder, ground-truth-npy folder).

---

## 4. Ground Truth & Density Map Specifications

Crowd counting is formulated as a continuous spatial regression problem rather than discrete object detection. Targets are continuous density maps stored as NPY (`.npy`) files.


* **Integral Count Property:** The total crowd count $C$ in any given image or region is computed mathematically via the integral (sum) of all pixel values within the density map:



$$C = \sum_{x,y} D(x,y)$$


* **Label Scaling Factor:** Density map pixel values generated via Gaussian kernels are exceptionally small, risking micro-gradient decay during backpropagation. Targets are multiplied by a scaling factor ($1000.0$) during training to ensure numerical stability, and divided by the same factor during evaluation metrics tracking.



---

## 5. Data Pipeline & Augmentation Strategy

The `CrowdDataModule` splits processing into joint spatial transformations and image-only photometric operations to preserve spatial alignment between RGB images and their corresponding density maps.

### Spatial Transformations (Jointly Applied)

* **Random Cropping:** During training, fixed-size spatial patches ($256 \times 256$) are randomly cropped simultaneously from both the RGB image and the density map. This mitigates GPU VRAM limitations and acts as a robust data augmentation technique.


* **Padding to Divisibility (`PadToMultiple(32)`):** Ensures spatial dimensions (Height and Width) are exact multiples of 32. This prevents shape mismatch errors during downsampling and skip-connection concatenations in backbones like decoders like U-Net and MAnet.


* **Random Horizontal Flipping ($p=0.5$):** Horizontally mirrors both the image and density map simultaneously, preserving coordinate synchronization and total count.



### Photometric Transformations (Image-Only)

* **Safe Photometric RandAugment:** Applies color jitter, solarization, and blurring strictly to the RGB image tensor. This injects severe training variance without mutating spatial coordinates or altering the density map's integral sum.


* **ImageNet Normalization:** Normalizes RGB pixel values using standard ImageNet distribution statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), while explicitly bypassing normalization masks on the density target.



---

<!-- ## 5. Dataset Acquisition & Conversion Workflow

Official crowd counting releases typically provide raw point annotations (`.mat` or JSON point coordinates) rather than pre-generated density maps, as optimal kernel sizing depends on network architecture.

1. Download raw point annotations from official repositories (e.g., ShanghaiTech or JHU-CROWD++ portals).


2. Execute conversion scripts utilizing geometry-adaptive Gaussian filters based on k-nearest neighbor ($k$-NN) average distances to generate smooth spatial distributions:



$$D(x) = \sum_{i=1}^{N} \delta(x - x_i) * G_{\sigma}(x)$$


3. Package and store the resulting arrays into standardized `.h5` files with naming conventions matching their corresponding source images. -->