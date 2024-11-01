import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'E:\RTDETR\RTDETR-20240703\RTDETR-main\runs\train\yolov8-detr-C2f-Faster-Rep-CAA-HSPAN\weights\best.pt') # select your model.pt path
    model.predict(source=r'E:\RTDETR\RTDETR-20240703\RTDETR-main\dataset\a',
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                  )