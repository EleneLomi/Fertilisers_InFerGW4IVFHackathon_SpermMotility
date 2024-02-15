def test_path_extraction():
    import pymotility.path_extraction as pe
    from skvideo.io import vread
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np

    root = "tests/data/videos"
    vid_names = [f"{root}/{name}" for name in os.listdir(root) if name.endswith(".mp4")]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"tests/data/path_extraction/{current}"
    os.mkdir(output_dir)
    videos = [vread(vid_name) for vid_name in vid_names]
    for method in pe.methods:
        os.mkdir(f"{output_dir}/{method}")
        for i, vid_name in enumerate(vid_names):
            path = pe.extract_path(videos[i], method=method, denoise=False)
            name = vid_name.split("/")[-1].split(".")[0]
            np.save(f"{output_dir}/{method}/{name}.npy", path)
            anim = pe.animate_path(videos[i], path)
            anim.save(f"{output_dir}/{method}/{name}.mp4")


if __name__ == "__main__":
    test_path_extraction()
