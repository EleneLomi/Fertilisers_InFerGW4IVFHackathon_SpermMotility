def test_segmentations():
    import pymotility.segmentation as seg
    import os
    from datetime import datetime

    root = "tests/data/videos"
    vid_paths = [f"{root}/{name}" for name in os.listdir(root)]
    current = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    output_dir = f"tests/data/segmentations/{current}"
    for method in seg.methods:
        method(vid_paths, output_dir)
        assert os.path.exists(output_dir)
