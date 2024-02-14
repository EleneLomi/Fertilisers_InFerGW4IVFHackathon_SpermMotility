import os
from datetime import datetime
import pymotility.path_extraction as pe
from skvideo.io import vread
import numpy as np


if __name__ == "__main__":
    root = "/Users/benn-m/Documents/ivf_hackathon/SpermDB/Sample3"
    vid_names = [
        f"{root}/{name}" for name in os.listdir(root) if name.endswith(".mp4")
    ]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"data/path_extraction/{current}"
    os.mkdir(output_dir)
    method = "dof"
    os.mkdir(f"{output_dir}/{method}")
    for i, vid_name in enumerate(vid_names):
        print(f"Processing {vid_name} ({i+1}/{len(vid_names)})")
        video = vread(vid_name)
        path = pe.extract_path(video, method=method, denoise=False)
        name = vid_name.split("/")[-1].split(".")[0]
        np.save(f"{output_dir}/{method}/{name}.npy", path)
