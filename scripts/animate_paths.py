import os
from datetime import datetime
import numpy as np
from skvideo.io import vread

import pymotility.path_extraction as pe


def animate_paths():
    root = "/Users/benn-m/Documents/ivf_hackathon/SpermDB"
    vid_names = [
        "sample3_vid7_sperm21_id345",
        "sample1_vid5_sperm15_id70",
        "sample1_vid1_sperm3_id3",
        "sample3_vid2_sperm13_id81",
        "sample3_vid9_sperm16_id149",
    ]
    vid_paths = []
    for vid_name in vid_names:
        if "sample1" in vid_name:
            vid_path = f"{root}/Sample1/{vid_name}.mp4"
        elif "sample3" in vid_name:
            vid_path = f"{root}/Sample3/{vid_name}.mp4"
        else:
            raise ValueError("Invalid video name")
        vid_paths.append(vid_path)
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"media/{current}"
    os.mkdir(output_dir)
    method = "lkof_framewise"
    for i, vid_name in enumerate(vid_paths):
        video = vread(vid_name)
        path = pe.extract_path(video, method=method, denoise=False)
        name = vid_name.split("/")[-1].split(".")[0]
        np.save(f"{output_dir}/{method}/{name}.npy", path)
        anim = pe.animate_path(video, path)
        anim.save(f"{output_dir}/{method}/{name}.mp4")


if __name__ == "__main__":
    animate_paths()
