# C-Transformer-UNet
This repository includes the official project of C-TransUNet, presented in our paper: Automated High-Resolution 3D Crevasse Mapping and Dynamic Linkages: An Integrated UAV-LiDAR, Photogrammetry, and C-TransUNet Framework

This repo holds code for C-TransUNet

📰 (News)
[June 2025] The initial draft of the paper has been completed, and the code is currently being organized and will soon be open-sourced.
[Future Outlook] Our model code and detailed workflow will be fully disclosed in this repository after the paper is accepted.

✨ Core Features
Innovative C-TransUNet Model: A deep learning architecture designed for precise segmentation of ice cracks, it combines the global context modeling capabilities of Transformers with the local feature extraction advantages of CNNs, achieving outstanding performance in ice crack extraction tasks (mIOU=88.04%, F1-Score=87.06%).
Multimodal Data Fusion Workflow: A novel workflow has been established that combines deep learning segmentation results with UAV-LS point cloud data, enabling systematic and automated extraction of three-dimensional geometric parameters (length, width, direction, depth) of ice crevasses.
Strong spatial generalization capability: The model demonstrates good spatial transferability and generalization capability on untrained glacier datasets (such as the WeigeleDangxiong Glacier and Xirilongpu Glacier).

🚀 Usage
1. Environment
Please ensure that Python 3.7+ is installed in your environment. Then use the following command to install the required dependencies.
We recommend installing them in a virtual environment.

Install all dependencies from the requirements.txt file.
pip install -r requirements.txt
Key dependency libraries include: PyTorch, NumPy, GDAL, rasterio, laspy, etc.

2. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
* 
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz

3 Prepare Data (The original drone imagery and LiDAR point cloud data used in this study can be obtained by contacting the corresponding author (wukunpeng@ynu.edu.cn) upon reasonable request.)

data/
└── yanong_glacier/
    ├── images/           # Store orthophotos of UAVs used for training and testing
    │   ├── train/
    │   └── test/
    ├── masks/            # Store corresponding ice crack labels (binary mask images)
    │   ├── train/
    │   └── test/
    └── point_clouds/     # Store UAV-LS point cloud data
        └── yanong_terminus.las

4 Train
# Running training scripts on a single GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset Yanong \
    --root_path ./data/yanong_glacier/ \
    --model_name C-TransUNet \
    --vit_name R50-ViT-B_16 \
    --vit_patches_size 16 \
    --max_epochs 350 \
    --batch_size 8 \
    --base_lr 0.01

5 Inference & Test

# Push all images in the test set folder
python inference.py \
    --model_name C-TransUNet \
    --vit_name R50-ViT-B_16 \
    --model_path /path/to/your/trained_model.pth \
    --input_dir ./data/yanong_glacier/images/test/ \
    --output_dir ./results/masks/

6 3D Parameter Extraction
This is what makes our framework unique.
After generating a two-dimensional mask image, run the following script in combination with LiDAR point cloud data to calculate the three-dimensional geometric parameters of the ice fissures.

python extract_3d_params.py \
    --mask_dir ./results/masks/ \
    --point_cloud ./data/yanong_glacier/point_clouds/yanong_terminus.las \
    --output_csv ./results/crevasse_3d_geometry.csv
    

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [ TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.](https://github.com/Beckschen)

Citations
