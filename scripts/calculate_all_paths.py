import os
from datetime import datetime

import numpy as np
from skvideo.io import vread

import pymotility.path_extraction as pe

if __name__ == "__main__":
    root = "/Users/benn-m/Documents/ivf_hackathon/SpermDB/Sample1"
    vid_names = [f"{root}/{name}" for name in os.listdir(root) if name.endswith(".mp4")]
    ommit = []
    for vid in ommit:
        try:
            vid_names.remove(f"{root}/{vid}.mp4")
        except ValueError:
            print(f"Could not remove {vid}.mp4")

    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"data/path_extraction/{current}"
    os.mkdir(output_dir)
    method = "lkof_framewise"
    os.mkdir(f"{output_dir}/{method}")
    for i, vid_name in enumerate(vid_names):
        print(f"Processing {vid_name} ({i+1}/{len(vid_names)})")
        video = vread(vid_name)
        path = pe.extract_path(video, method=method, denoise=False)
        name = vid_name.split("/")[-1].split(".")[0]
        np.save(f"{output_dir}/{method}/{name}.npy", path)
