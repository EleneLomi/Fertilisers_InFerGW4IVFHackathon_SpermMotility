import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skvideo.io import vread

import pymotility.path_extraction as pe

plt.style.use("ggplot")


def load_hand_tracked_path(name):
    x_data = np.array(pd.read_csv(name))[:, 1::2]
    y_data = np.array(pd.read_csv(name))[:, 2::2]
    T = x_data.shape[0]
    x_data.shape[1]
    diffs = np.ones((T, 2)) * np.nan
    diffs[0, :] = [0, 0]
    for i in range(1, T - 1):
        diffs[i, 0] = np.nanmean(x_data[i + 1, :] - x_data[i, :])
        diffs[i, 1] = np.nanmean(y_data[i + 1, :] - y_data[i, :])
    path = np.cumsum(diffs, axis=0)
    return path


def test_path_extraction():
    root = "tests/data/videos"
    vid_names = [f"{root}/{name}" for name in os.listdir(root) if name.endswith(".mp4")]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"tests/data/path_extraction/{current}"
    os.mkdir(output_dir)
    for method in pe.methods:
        os.mkdir(f"{output_dir}/{method}")
        for i, vid_name in enumerate(vid_names):
            video = vread(vid_name)
            path = pe.extract_path(video, method=method, denoise=False)
            name = vid_name.split("/")[-1].split(".")[0]
            np.save(f"{output_dir}/{method}/{name}.npy", path)
            anim = pe.animate_path(video, path)
            anim.save(f"{output_dir}/{method}/{name}.mp4")


def test_hand_tracked_paths():
    vid_path = "tests/data/videos"
    csv_path = "tests/data/tracked_videos"
    csv_names = [name for name in os.listdir(csv_path) if name.endswith(".csv")]
    vid_names = [f"{vid_path}/{name.split('.')[0]}.mp4" for name in csv_names]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"tests/data/path_extraction/{current}"
    os.mkdir(output_dir)
    for method in pe.methods:
        os.mkdir(f"{output_dir}/{method}")
        for i, vid_name in enumerate(vid_names):
            video = vread(vid_name)
            path = pe.extract_path(video, method=method, denoise=False)
            ht_path = load_hand_tracked_path(f"{csv_path}/{csv_names[i]}")
            # get the last non nan index in ht_path
            last = np.where(np.isnan(ht_path[:, 0]))[0][0]
            fig, ax = plt.subplots()
            ax.plot(path[:last, 0], path[:last, 1], label="Extracted")
            ax.plot(ht_path[:last, 0], ht_path[:last, 1], label="Hand Tracked")
            error = np.linalg.norm(path[:last, :] - ht_path[:last, :], axis=1)
            avg_error = np.mean(error)
            # add legend containing the average error
            ax.legend(title=f"Average Error: {avg_error:.2f}")
            ax.set(title=f"Method: {method}. Video: {vid_name.split('/')[-1].split('.')[0]}")
            plt.savefig(f"{output_dir}/{method}/{csv_names[i].split('.')[0]}.png")


if __name__ == "__main__":
    # test_path_extraction()
    test_hand_tracked_paths()
