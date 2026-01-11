from ultralytics import YOLO
from pathlib import Path
def main():

    runs_dir = Path("runs/classify")     
    
    train_dirs = sorted(
        [p for p in runs_dir.glob("train*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    traget_checkpoint = None

    for p in train_dirs:
        checkpoint = p / "weights" / "last.pt"
        if checkpoint.exists():
            traget_checkpoint = checkpoint
            print(f"Resuming from checkpoint: {traget_checkpoint}")
            break
    
    if traget_checkpoint:
        print("Loading model...")
        try:
            model = YOLO(traget_checkpoint)
            model.train(resume=True)
        except Exception as e:
            print(f"Failed to resume training from checkpoint: {e}")
    else:
        print("No checkpoint found to resume from.")

    metrics = model.val()
    print(metrics.results_dict)

if __name__=="__main__":
    main()