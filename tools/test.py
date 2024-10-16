import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

model = YOLO("yolov8n-mlca.yaml")  # build a new model from scratch
