# MIRAGE
The code repo for paper ``MIRAGE: Biomechanically-Interpretable Maize Plant 3D Generation and Single-view Reconstruction``


## Requirements
This project is designed for **Python 3.10** and depends on the following core packages:

``pip install torch numpy matplotlib geomdl trimesh MinkowskiEngine pytorch3d scipy open3d pytorch-lightning wandb``

## Usage
``approx_ctrlpts.zip``: the approxed and processed control points of the leaf meshes original provided in https://huggingface.co/datasets/BGLab/MaizeField3D
``maize_voxel.zip``: the voxelized 3D maize model with sementic labels.
``augment_pipeline.py``: the code for generating 3D mesh-based maize based on the proposed augmentation technique, used for the subsequent model training.
``train.py``: the code for training maize reconstruction model from single-view
``train_unconditioned.py``: the code for training unconditional maize 3D generation model
