import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'PFLO.yaml')
    model.train(data=r"data\9k9\coco-pose.yaml",  # 训练数据集路径
                imgsz=1280,
                task='pose',
                epochs=130,
                batch=4,
                )