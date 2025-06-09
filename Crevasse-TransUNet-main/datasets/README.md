# Data Preparation for C-TransUNet (VOC Format)

This document provides detailed instructions for preparing a custom glacier crevasse dataset in the **PASCAL VOC format** for use with the C-TransUNet model.

### 1. Required Directory Structure

Your dataset must follow the structure below. It is recommended to create a `VOCdevkit` folder in the root of the project.

```bash
<project_root>/
└── VOCdevkit/
    └── VOC2007/
        ├── ImageSets/
        │   └── Segmentation/
        │       ├── train.txt
        │       ├── val.txt   # (Optional, for validation)
        │       └── test.txt
        ├── JPEGImages/
        │   ├── image_0001.jpg
        │   ├── image_0002.jpg
        │   └── ...
        └── SegmentationClass/
            ├── image_0001.png
            ├── image_0002.png
            └── ...
```

### 2. Step-by-Step Guide

#### Step 2.1: Populate Image and Mask Folders

1.  **`JPEGImages`**:
    - Place all your original images in this folder.
    - Images must be in `.jpg` format.

2.  **`SegmentationClass`**:
    - Place all your corresponding label masks in this folder.
    - The filename for a mask **must** exactly match the basename of its corresponding image (e.g., `image_0001.jpg` corresponds to `image_0001.png`).
    - Masks must be single-channel, 8-bit `.png` files.
    - **Crucially, the pixel values in the masks must represent the class index:**
        - **`0`**: for the background class.
        - **`1`**: for the crevasse (foreground) class.

    You can use the script you provided to convert your original masks (e.g., with pixel values 0 and 255) to this required format (0 and 1).

#### Step 2.2: Create File Lists for Data Splits

The model needs to know which images belong to the training, validation, and testing sets. You must create `.txt` files in the `ImageSets/Segmentation/` directory to define these splits.

-   **`train.txt`**: Contains the list of filenames for the **training set**.
-   **`val.txt`**: Contains the list of filenames for the **validation set**.
-   **`test.txt`**: Contains the list of filenames for the **testing set**.

**Format**: Each file should list the basenames of the image files (without the `.jpg` or `.png` extension), with one name per line.

**Example `train.txt`:**
```
image_0001
image_0003
image_0005
...
```

**Example `test.txt`:**
```
image_0002
image_0004
...
```

Once you have completed these steps, your dataset will be correctly formatted and ready for use with the provided training and testing scripts.