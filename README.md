# PFLO: A High-Throughput Pose Estimation Model for Field Maize based on YOLO Architecture

PFLO is a high-throughput pose estimation model for field maize based on the YOLO architecture, specifically designed for maize in field environments. This model incorporates multiple structural optimizations to accurately estimate maize posture under challenging conditions including complex backgrounds, dense planting, occlusions, and morphological variations.

The dataset in this study can be accessed at: http://phenomics.agis.org.cn/#/category.

## Install

```bash
# Clone
git clone https://github.com/Akacaesarp/PFLO.git
cd PFLO
# Install requirements
pip install ultralytics

```
## Dataset Preparation
```bash
maize pose dataset
├── images
│   ├── train
│   ├── val
│   └── test
└── labels
│   ├── train
│   ├── val
│   └── test
├── Maize_pose.yaml
```

## Training

```bash
# Start training# Start training
python train.py --data dataset/Maize_pose.yaml --batch 4 --epochs 130 --img 1280 --device 0
```
or you can just run
```bash

from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'PFLO.yaml')
    model.train(data=r"Maize_pose.yaml",
                imgsz=1280,
                task='pose',
                epochs=130,
                batch=4,
                )
```
## Inference
```bash
# Run inference on an image
python predict.py --weights weights/PFLO.pt --source /path/to/image.jpg --save-txt --save-conf

# Run inference on a directory of images
python predict.py --weights weights/PFLO.pt --source /path/to/image/folder --save-txt --save-conf
```
