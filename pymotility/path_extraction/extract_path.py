import numpy as np
from skvideo.io import vread, vwrite
import cv2
import os

methods = ["lkof", "trex"]


def relight(video):
    """Relight the video using a uniform convolution."""
    pass


def denoise(video):
    """Denoise the video."""
    pass


def extract_path(video, method="lkof", denoise=False, relight=False):
    """extract_path the video using the specified method."""
    if isinstance(video, list):
        return [
            extract_path(video, method, denoise, relight) for video in video
        ]
    if isinstance(video, str):
        video = vread(video)
    if relight:
        video = relight(video)
    if denoise:
        video = denoise(video)
    if method == "lkof":
        path = lkof_extract_path(video)
    elif method == "trex":
        path = trex_extract_path(video)
    else:
        raise ValueError(f"Invalid method: {method}")
    return path


def lkof_extract_path(video):
    """extract_path the video using the Lucas Kanade Optical Flow method."""
    T, N, M, _ = video.shape
    # get the good points to track
    old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    path = np.full((T, 2), (N // 2, M // 2))
    return path


def trex_extract_path(video):
    """extract_path the video using the TREX method."""
    T, N, M, _ = video.shape
    path = np.full((T, 2), (N // 2, M // 2))
    return path
