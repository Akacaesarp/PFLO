# PFLO: A High-Throughput Pose Estimation Model for Field Maize based on YOLO Architecture

PFLO is a high-throughput pose estimation model for field maize based on the YOLO architecture, specifically designed for maize in field environments. This model incorporates multiple structural optimizations to accurately estimate maize posture under challenging conditions including complex backgrounds, dense planting, occlusions, and morphological variations.

The dataset in this study can be accessed at: http://phenomics.agis.org.cn/#/category.

In the MIPDB database, annotations and images are stored in JSON format. You need to use data_preprocess/data_preprocess.py to convert the JSON files to the YOLO format:
## Dataset Preparation

```bash
# Extract images from JSON files
python data_preprocess/data_preprocess.py --task extract --json-dir /path/to/json/files --save-dir /path/to/images

# Convert JSON annotations to YOLO format
python data_preprocess/data_preprocess.py --task convert --json-dir /path/to/json/files --save-dir /path/to/labels

# Process and interpolate keypoints
python data_preprocess/data_preprocess.py --task process --json-dir /path/to/json/files --save-dir /path/to/processed

# Or run all tasks at once
python data_preprocess/data_preprocess.py --task all --json-dir /path/to/json/files --save-dir /path/to/output --extract-dir /path/to/images --convert-dir /path/to/labels --process-dir /path/to/processed

```
## Download Weights
Download the PPF-YOLO.pt weights file from the [Releases](https://github.com/Akacaesarp/PFLO/releases)

## Install

```bash
# Clone
git clone https://github.com/Akacaesarp/PFLO.git
cd PFLO
# Install requirements
pip install ultralytics

```
## Dataset Format
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
or you can just run train.py：
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
python predict.py --weights weights/PFLO.pt --source /path/to/image.jpg 

# Run inference on a directory of images
python predict.py --weights weights/PFLO.pt --source /path/to/image/folder
```
or you can just run predict.py
```bash
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('PFLO.pt')
    result=model.predict(R"test_images", 
                save = True,
                show_boxes=False,
                show_labels=False,
                project='EXP1',
                name='Stage_R1'
                )
```
