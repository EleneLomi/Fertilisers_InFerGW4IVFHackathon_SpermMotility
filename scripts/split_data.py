import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    hand_labeled = pd.read_csv("data/path_extraction/ManualMotilityAnalysis.csv")
    test_names = hand_labeled["Video name"]
    test_names = [name + ".npy" for name in test_names]
    train = "data/training_data/train"
    test = "data/training_data/test"
    os.mkdir(train)
    os.mkdir(test)
    # split the labeled into the test folder
    roots = ["data/path_extraction/sample_1_paths/lkof_framewise", "data/path_extraction/sample_3_paths/lkof_framewise"]
    for root in roots:
        for name in os.listdir(root):
            if name in test_names:
                path = np.load(f"{root}/{name}")
                np.save(f"{test}/{name}", path)
            else:
                path = np.load(f"{root}/{name}")
                np.save(f"{train}/{name}", path)
