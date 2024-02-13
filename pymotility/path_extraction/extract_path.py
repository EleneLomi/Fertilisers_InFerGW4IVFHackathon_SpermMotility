import numpy as np
from skvideo.io import vread, vwrite
import cv2
import os
import matplotlib.pyplot as plt
import time

methods = ["lkof"]


def relight(video):
    """Relight the video using a uniform convolution."""
    pass


def denoise(video):
    """Denoise the video."""
    pass


def exclude_center_roi(width, height, N, M):
    exclude_center_roi = np.ones((N, M), dtype=np.uint8)
    exclude_center_roi[
        N // 2 - width : N // 2 + width, M // 2 - height : M // 2 + height
    ] = 0
    exclude_center_roi = exclude_center_roi.astype(np.uint8)
    return exclude_center_roi


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
    else:
        raise ValueError(f"Invalid method: {method}")
    return path


def lkof_extract_path(video):
    """extract_path the video using the Lucas Kanade Optical Flow method."""
    T, N, M, _ = video.shape
    ql = 0.2
    path = np.zeros((T, 2))
    tracked_paths = []
    old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
    ecr = exclude_center_roi(N // 10, N // 10, N, M)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    points = cv2.goodFeaturesToTrack(
        old_gray, mask=ecr, maxCorners=100, qualityLevel=ql, minDistance=70
    )
    trajectories = np.zeros((T, points.shape[0], 2))
    fig, ax = plt.subplots()
    for i, frame in enumerate(video):
        trajectories[i] = points.reshape(-1, 2)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, points, None, **lk_params
        )
        points[st == 0] = np.nan
        if np.sum(~np.isnan(points)) < 2:
            new_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=ecr,
                maxCorners=100,
                qualityLevel=ql,
                minDistance=70,
            )
            points = np.vstack([points, new_points])
            trajectories = np.concatenate(
                [trajectories, np.nan * np.ones((T, new_points.shape[0], 2))],
                axis=1,
            )
    return path
