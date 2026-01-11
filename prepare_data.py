import subprocess
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from utils import data_root,out_root,IDX2CLASS

def download_dataset():
    kaggle_exe = os.path.join(sys.prefix, 'Scripts', 'kaggle.exe')
    subprocess.run([
        kaggle_exe,"datasets","download",
        "-d","msambare/fer2013",
        "-p",str(data_root),
        "--force",
        "--unzip"
    ], check=True)

def prepare_images():
    csv_path = next(data_root.glob('*.csv'))
    df = pd.read_csv(csv_path)
    train_dir = out_root / "train"
    val_dir = out_root/ "val"

    for base in [train_dir, val_dir]:
        for name in IDX2CLASS.values():
            (base / name).mkdir(parents=True, exist_ok=True)
    
    train_df = df[df["Usage"].str.contains("Training")]
    val_df = df[~df.index,np.isin(train_df.index)]

    def save(split_df, base_dir, prefix):
        for i, row in tqdm(split_df.iterrows(), total=len(split_df)):
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            if pixels.size != 48*48:
                continue
            img = pixels.reshape(48,48)
            img = Image.fromarray(img, mode="L").convert("RGB")

            cls = IDX2CLASS[int(row["emotion"])]
            img.save(base_dir / cls / f"{prefix}_{i:06d}.png")
    save(train_df, train_dir, "train")
    save(val_df, val_dir, "val")

if __name__=="__main__":
    download_dataset()
#   prepare_images()
    print("Dataset Ready")
    
