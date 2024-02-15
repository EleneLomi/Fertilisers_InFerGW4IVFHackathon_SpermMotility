import numpy as np
import numpy.linalg as la
from skvideo.io import vread, vwrite
import cv2
import os
import matplotlib.pyplot as plt
import time

methods = ["lkof_framewise", "dof"]


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
        return [extract_path(video, method, denoise, relight) for video in video]
    if isinstance(video, str):
        video = vread(video)
    # video = grayscale_video(video)
    if relight:
        video = relight_video(video)
    if denoise:
        video = denoise_video(video)
    if method == "lkof":
        path = lkof_extract_path(video)
    elif method == "lkof_framewise":
        path = lkof_framewise_extract_path(video)
    elif method == "dof":
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


def lkof_framewise_extract_path(video):
    T, N, M, _ = video.shape
    bql = 0.1
    old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
    ecr = exclude_center_roi(N // 10, N // 10, N, M)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    points = []
    path = np.zeros((T, 2))
    for i, frame in enumerate(video[1:]):
        ql = bql
        while len(points) < 10:
            points = cv2.goodFeaturesToTrack(
                old_gray,
                mask=ecr,
                maxCorners=100,
                qualityLevel=ql,
                minDistance=70,
            )
            ql *= 0.9
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, points, None, **lk_params
        )
        old_gray = frame_gray.copy()
        diffs = new_points - points
        diffs = diffs[st == 1]
        # remove outliers
        centered = diffs - np.median(diffs, axis=0)
        inverse_covariance_matrix = la.inv(np.cov(centered.T))
        mahalanobis = np.array(
            [np.dot(dist, np.dot(inverse_covariance_matrix, dist)) for dist in centered]
        )
        diffs = diffs[mahalanobis < 3]
        if np.all(np.isnan(diffs)):
            path[i + 1] = path[i]
        else:
            path[i + 1] = path[i] + np.mean(diffs, axis=0)
    return path


def dof_extract_path(video, verbose=True):
    """Extract path from the video using the Dense Optical Flow method."""
    T, N, M, _ = video.shape
    thetas = np.zeros(T - 1)
    dists = np.zeros(T - 1)
    path = np.zeros((T, 2))
    path[0] = np.array([0, 0])
    last = video[0]
    hsv = np.zeros_like(last)
    hsv[..., 1] = 255
    last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    if verbose:
        print("Extracting Path")
    for i, frame in enumerate(video[1:]):
        if i % 50 == 0 and verbose:
            print(f"Processing frame {i}/{T}")
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(last, next, None, 0.5, 11, 15, 3, 5, 1.2, 0)
        last = next.copy()
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ecr = exclude_center_roi(N // 10, N // 10, N, M)
        # remove not tracked background (bodge)
        mag_max = mag[np.where(ecr)].max()
        ret, thresh = cv2.threshold(mag, 3 * mag_max / 4, mag_max, cv2.THRESH_BINARY)
        mask = ecr & thresh.astype(int)
        if np.unique(mask).size == 1:
            thetas[i] = 0
            dists[i] = 0
            path[i + 1] = path[i]
        else:
            thetas[i] = np.mean(ang[np.where(mask)])
            dists[i] = np.mean(mag[np.where(mask)])
            path[i + 1] = path[i] + dists[i] * np.array(
                [np.cos(thetas[i]), np.sin(thetas[i])]
            )
    return path
