import numpy as np
from skvideo.io import vread, vwrite
import cv2
import os
import matplotlib.pyplot as plt
import time

methods = ["dof", "lkof"]


def grayscale_video(video):
    """Convert the video to greyscale."""
    print("Converting video to greyscale")
    out = video[..., 0].copy()
    for i, frame in enumerate(video):
        out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Conversion complete")
    return out


def relight_video(video):
    """Relight the video using a uniform convolution."""
    pass


def denoise_video(video):
    """Denoise a greyscale video."""
    print("Denoising video")
    for i, frame in enumerate(video):
        video[i] = cv2.fastNlMeansDenoising(
            frame,
            None,
            templateWindowSize=7,
            searchWindowSize=15,
            h=1,
        )
        # cv2.imshow("frame", video[i])
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27:
        #     break
    print("Denoising complete")
    return video


def exclude_center_roi(width, height, N, M):
    exclude_center_roi = np.ones((N, M), dtype=np.uint8)
    exclude_center_roi[
        N // 2 - width : N // 2 + width, M // 2 - height : M // 2 + height
    ] = 0
    exclude_center_roi = exclude_center_roi.astype(np.uint8)
    return exclude_center_roi


def extract_path(video, method="dof", denoise=False, relight=False):
    """extract_path the video using the specified method."""
    if isinstance(video, list):
        return [
            extract_path(video, method, denoise, relight) for video in video
        ]
    if isinstance(video, str):
        video = vread(video)
    video = grayscale_video(video)
    if relight:
        video = relight_video(video)
    if denoise:
        video = denoise_video(video)
    if method == "lkof":
        path = lkof_extract_path(video)
    if method == "dof":
        path = dof_extract_path(video)
    else:
        raise ValueError(f"Invalid method: {method}")
    return path


def lkof_extract_path(video):
    """extract_path the video using the Lucas Kanade Optical Flow method."""
    T, N, M = video.shape
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


def dof_extract_path(video):
    """Extract path from the video using the Dense Optical Flow method."""
    T, N, M = video.shape
    last = video[0]
    hsv = np.zeros((N, M, 3), dtype=np.float16)
    hsv[..., 1] = 255
    for i, frame in enumerate(video[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            last, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        last = frame.copy()
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv.astype(int), cv2.COLOR_HSV2BGR)
        cv2.imshow("frame2", bgr)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    return np.zeros((T, 2))
