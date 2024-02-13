import numpy as np
from skvideo.io import vread, vwrite
import os

methods = ["lkof", "trex"]


def relight(video):
    """Relight the video using a uniform convolution."""
    pass


def denoise(video):
    """Denoise the video."""
    pass


def segment(video, method="lkof", denoise=False, relight=False):
    """Segment the video using the specified method."""
    if isinstance(video, list):
        return [segment(video, method, denoise, relight) for video in video]
    if isinstance(video, str):
        video = vread(video)
    if relight:
        video = relight(video)
    if denoise:
        video = denoise(video)
    if method == "lkof":
        path = lkof_segment(video)
    elif method == "trex":
        path = trex_segment(video)
    else:
        raise ValueError(f"Invalid method: {method}")
    return path


def lkof_segment(video):
    """Segment the video using the Lucas Kanade Optical Flow method."""
    T, N, M, _ = video.shape
    path = np.zeros((T, 2))
    return path


def trex_segment(video):
    """Segment the video using the TREX method."""
    T, N, M, _ = video.shape
    path = np.zeros((T, 2))
    return path
