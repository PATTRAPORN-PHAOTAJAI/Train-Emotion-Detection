from ultralytics import YOLO
from utils import out_root
import os

def main():
    model = YOLO("yolo11n-cls.pt")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    print(f"Training with data from: {data_path}")
    model.train(
        data=data_path,
        epochs=2,
        imgsz=64,
        batch=64
    )

if __name__=="__main__":
    main()
