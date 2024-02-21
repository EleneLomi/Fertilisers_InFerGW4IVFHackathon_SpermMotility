import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from skvideo.io import vread

methods = ["lkof_framewise", "dof"]
plt.style.use("ggplot")


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
    exclude_center_roi[N // 2 - width : N // 2 + width, M // 2 - height : M // 2 + height] = 0
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
    old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
    ecr = exclude_center_roi(N // 10, N // 10, N, M)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    points = cv2.goodFeaturesToTrack(old_gray, mask=ecr, maxCorners=100, qualityLevel=ql, minDistance=70)
    trajectories = np.zeros((T, points.shape[0], 2))
    fig, ax = plt.subplots()
    for i, frame in enumerate(video):
        trajectories[i] = points.reshape(-1, 2)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
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


def lkof_framewise_extract_path(video, show_outliers=False):
    T, N, M, _ = video.shape
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
        points = cv2.goodFeaturesToTrack(
            old_gray,
            mask=ecr,
            maxCorners=20,
            qualityLevel=1e-2,
            minDistance=50,
        )
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
        old_gray = frame_gray.copy()
        diffs = new_points - points
        diffs = diffs[st == 1]
        # remove outliers
        centered = diffs - np.mean(diffs, axis=0)
        try:
            inverse_covariance_matrix = la.inv(np.cov(centered.T))
        except la.LinAlgError:
            # use ecuclidean distance if covariance matrix is singular
            std = np.std(la.norm(centered, axis=0))
            inverse_covariance_matrix = np.eye(2) / std**2
        mahalanobis = np.array([np.sqrt(np.dot(dist, np.dot(inverse_covariance_matrix, dist))) for dist in centered])
        thresh = 2
        if np.any(mahalanobis > thresh) and show_outliers:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(10, 5)
            fig.suptitle("Tracked Points")
            ax[0].imshow(video[i - 1, :])
            in_points = np.where(mahalanobis < thresh)
            out_points = np.where(mahalanobis > thresh)
            mean = np.mean(diffs[in_points], axis=0)
            ax[0].quiver(
                points[in_points, 0, 0],
                points[in_points, 0, 1],
                diffs[in_points, 0],
                diffs[in_points, 1],
                angles="xy",
                scale_units="xy",
                scale=0.05,
                color="g",
                label="Inliers",
            )
            ax[0].quiver(
                points[out_points, 0, 0],
                points[out_points, 0, 1],
                diffs[out_points, 0],
                diffs[out_points, 1],
                angles="xy",
                scale_units="xy",
                scale=0.05,
                color="r",
                label="Outliers",
            )
            ax[0].quiver(
                N // 2, M // 2, mean[0], mean[1], angles="xy", scale_units="xy", scale=0.05, color="k", label="Mean"
            )
            ax[0].legend()
            ax[0].grid(False)
            ax[0].axis("off")
            ax[0].set_title("Optical Flow Vectors")
            ax[1].set_title("Optical Flow Vectors")
            s = ax[1].scatter(diffs[:, 0], -diffs[:, 1], c=mahalanobis, cmap="viridis")
            colors = s.to_rgba(mahalanobis)
            zs = np.zeros_like(diffs[:, 0])
            q = ax[1].quiver(zs, zs, diffs[:, 0], -diffs[:, 1], color=colors, angles="xy", scale_units="xy", scale=1)
            ax[1].quiver(0, 0, mean[0], -mean[1], color="k")
            ax[1].axis("equal")
            length = np.max(np.abs(diffs))
            ax[1].set(xlim=(-length, length), ylim=(-length, length))
            cb = plt.colorbar(s, ax=ax[1])
            cb.set_label("Mahalanobis Distance")
            plt.show()

        diffs = diffs[mahalanobis < 2]
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
            path[i + 1] = path[i] + dists[i] * np.array([np.cos(thetas[i]), np.sin(thetas[i])])
    return path
