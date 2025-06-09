# C-TransUNet for Glacier Crevasse Segmentation

This repository contains a PyTorch implementation of the **C-TransUNet** model, as described in the paper "[Automated High-Resolution 3D Crevasse Mapping and Dynamic Linkages...](https://doi.org/10.1016/j.jag.2025.104495)". The codebase is adapted from the original [TransUNet](https://github.com/Beckschen/TransUNet) implementation to specifically address the task of segmenting glacier crevasses from high-resolution UAV imagery.

## Model Architecture

C-TransUNet is a hybrid Convolutional Neural Network (CNN) and Transformer model designed for precise semantic segmentation. It leverages the strengths of both architectures to achieve a superior balance between local detail preservation and global semantic understanding.

The core architecture consists of:
- **Hybrid Encoder**: A ResNet-50 backbone acts as a powerful feature extractor, providing robust local feature maps at multiple scales. These features are then passed to a Vision Transformer (ViT) encoder to model long-range dependencies and global context across the entire image.
- **Key Innovations**: This implementation incorporates the main contributions of the C-TransUNet paper:
    1.  **CNN-Guided Positional Encoding**: An innovative correction term, derived from the ResNet features, is added to the Transformer's position embeddings to dynamically fuse spatial location information with locally extracted texture features.
    2.  **Adaptive Multi-Scale Feature Fusion**: The standard U-Net skip-connections are enhanced with a dual attention gating mechanism. This allows the decoder to adaptively control the flow of information from the encoder, selectively emphasizing informative features and improving the localization accuracy of crevasse edges.
- **Decoder**: A cascaded upsampler (CUP) path, similar to U-Net, progressively restores the feature map resolution, generating a precise, full-resolution segmentation mask.

## Directory Structure

```bash
.
├── VOCdevkit
│   └── VOC2007
│       ├── ImageSets
│       │   └── Segmentation
│       │       ├── train.txt
│       │       ├── val.txt
│       │       └── test.txt
│       ├── JPEGImages
│       │   ├── image1.jpg
│       │   └── ...
│       └── SegmentationClass
│           ├── image1.png
│           └── ...
├── model
│   ├── vit_checkpoint
│   │   └── imagenet21k
│   │       └── R50+ViT-B_16.npz
│   └── C-TransUNet_Crevasse_... (Saved model snapshots)
├── networks
├── datasets
└── ... (train.py, test.py, etc.)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/C-TransUNet-Crevasse.git](https://github.com/YourUsername/C-TransUNet-Crevasse.git)
    cd C-TransUNet-Crevasse
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The model expects data to be structured in the **PASCAL VOC** format. Detailed instructions for preparing the dataset are provided in [`datasets/README.md`](./datasets/README.md).

The general workflow is:
1.  Create a `VOCdevkit/VOC2007` directory in the project root.
2.  Place all your `.jpg` training and testing images into `VOCdevkit/VOC2007/JPEGImages/`.
3.  Place all your corresponding `.png` binary label masks into `VOCdevkit/VOC2007/SegmentationClass/`. The masks should have pixel values of `0` for background and `1` for crevasses.
4.  Create `train.txt`, `val.txt`, and `test.txt` files inside `VOCdevkit/VOC2007/ImageSets/Segmentation/`. These files should contain the basenames of the images for each split (without file extensions).

## Training

To train the C-TransUNet model, run the `train.py` script. The `--data_path` argument should point to your `VOCdevkit/VOC2007` directory.

```bash
python train.py --dataset Crevasse --data_path ./VOCdevkit/VOC2007 --img_size 512 --batch_size 8 --base_lr 0.001 --max_epochs 350
```

- `--data_path`: Path to the root of the VOC-formatted dataset.
- `--img_size`: Input image size, 512 is recommended as per the paper.
- `--batch_size`: Adjust based on your GPU memory. `512x512` images require significant VRAM.
- `--max_epochs`: The paper suggests a training duration of around 350 epochs.
- Trained models will be saved to a uniquely named folder inside the `model/` directory (note: the code uses `../model`, so it will be outside the project folder by default).

## Testing & Inference

To evaluate a trained model, run the `test.py` script. The script automatically constructs the path to the corresponding trained model based on the provided arguments, so ensure they match the training run.

```bash
python test.py --dataset Crevasse --data_path ./VOCdevkit/VOC2007 --img_size 512 --is_savenii
```

- `--data_path`: Path to the directory containing the test set, same as for training. The script will use `test.txt` from this directory.
- `--is_savenii`: If this flag is used, the model's predictions (as PNGs) will be saved in the `predictions` directory, which is useful for visualization and further analysis.

## Citation

If you use this work, please cite the relevant papers:

**C-TransUNet (The model this repository implements):**
```bibtex

@article{duan2025,
  title={Automated High-Resolution 3D Crevasse Mapping and Dynamic Linkages: An Integrated UAV-LiDAR, Photogrammetry, and C-TransUNet Framework},
  author={Duan, Yunpeng and Wu, Kunpeng and Liu, Shirin and Zhou, Jun and Yang, Xin and Gao, Daoxun},
  journal={...}, 
  volume={...},
  pages={...},
  year={...},
  publisher={...}
}
```

**Original TransUNet:**
```bibtex
@article{chen2021transunet,
  title={Transunet: Transformers make strong encoders for medical image segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```