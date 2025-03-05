import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('PFLO.pt')
    result=model.predict(R"test_images",  # 训练数据集路径
                save = True,
                show_boxes=False,
                show_labels=False,
                project='EXP1',
                name='R1'
                )
    a=1