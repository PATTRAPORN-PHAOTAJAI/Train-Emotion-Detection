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
        epochs=300,
        imgsz=64,
        batch=64,
        patience=50,
        device="0",

    )
    metrics = model.val()
    print(metrics.results_dict)


if __name__=="__main__":
    main()
