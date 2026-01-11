from pathlib import Path

project_root = Path(__file__).parent

data_root = project_root / "data"
out_root = project_root / "fer_yolo_cls"
modle_dir = project_root / "models"

data_root.mkdir(exist_ok=True)
out_root.mkdir(exist_ok=True)
modle_dir.mkdir(exist_ok=True)

IDX2CLASS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "suprise",
    6: "neutral"
}

