def test_segmentations():
    import pymotility.segmentation as seg
    import os
    from datetime import datetime
    import numpy as np

    root = "tests/data/videos"
    vid_names = [
        f"{root}/{name}" for name in os.listdir(root) if name.endswith(".mp4")
    ]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"tests/data/segmentations/{current}"
    os.mkdir(output_dir)
    for method in seg.methods:
        os.mkdir(f"{output_dir}/{method}")
        paths = seg.segment(vid_names, method=method)
        for path, vid_name in zip(paths, vid_names):
            name = vid_name.split("/")[-1].split(".")[0]
            np.save(f"{output_dir}/{method}/{name}.npy", path)


if __name__ == "__main__":
    test_segmentations()
