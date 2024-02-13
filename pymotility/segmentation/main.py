import numpy as np
from skvideo.io import vread, vwrite

methods = ["lkof", "trex"]


def relight(video):
    """Relight the video using a uniform convolution."""
    pass


def denoise(video):
    """Denoise the video."""
    pass


def segment(video, method, denoise=True, relight=True):
    """Segment the video using the specified method."""
    if video.isinstance(list):
        return [segment(video, method, denoise, relight) for video in video]
    if video.isinstance(str):
        video = vread(video)
    if method == "lkof":
        return lkof_segment(video)
    if method == "trex":
        return trex_segment(video)
    raise ValueError(f"Invalid method: {method}")


def lkof_segment(video):
    """Segment the video using the Lucas Kanade Optical Flow method."""
    T, N, M = video.shape
    path = np.zeros((T, 2))
    return path


def trex_segment(video):
    """Segment the video using the TREX method."""
    T, N, M = video.shape
    path = np.zeros((T, 2))
    return path
