import json
from pathlib import Path
from ultralytics import YOLO
from utils import modle_dir

def main():
    runs = Path("runs/classify")
    train_dirs = sorted(
        [p for p in runs.glob("train*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    best_pt = train_dirs[0] / "weights" / "best.pt"
    model = YOLO(best_pt)
    model.export(format="onnx", opset=12)
    
    onnx_path = next(best_pt.parent.glob("*.onnx"))
    final_onnx = modle_dir / "emotion_yolo.onnx"
    onnx_path.replace(final_onnx)
    
    data_train_path = Path("data/train")
    classes = sorted([p.name for p in data_train_path.iterdir()])
    with open(modle_dir / "classes.json", "w" , encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)
    
    latest_train_dir = train_dirs[0] 
    print(f"Pulling the model from the latest round.: {latest_train_dir}")
    print("Exported ONNX model:" , final_onnx)
    print(f"ONNX model and classes saved to {modle_dir}")

if __name__=="__main__":
    main()
