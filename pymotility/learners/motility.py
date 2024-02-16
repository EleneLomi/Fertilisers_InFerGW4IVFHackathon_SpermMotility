import numpy as np

"""
    Metric from "A robust sperm tracking algorithm ..." Alabdulla et ad.
    assumes frame rate is 1
    assumes path is of shape (T, 2) where T is the number of time steps
"""


# VCL
def culvilinear_velocity(path):
    diffs = np.diff(path, axis=0)
    squared_diffs = np.square(diffs)
    distances = np.sqrt(np.sum(squared_diffs, axis=1))
    vcl = 1 / (path.shape[0] - 1) * np.sum(distances)
    return vcl


# VSL
def straight_line_velocity(path):
    straigth_line_distance = np.sqrt(np.sum(np.square(path[-1] - path[0])))
    return straigth_line_distance / (path.shape[0] - 1)


def smooth_path(line):
    return sum(line) / len(line)


# VAP
def average_line_velocity(path, smoothing_window=5):
    smoothed_path = [
        smooth_path(path[i : i + smoothing_window])
        for i in range(0, len(path) - smoothing_window)
    ]
    # turn into np.array
    smoothed_path = np.array(smoothed_path)
    return straight_line_velocity(smoothed_path)


# LIN
def linearity_progressive_motility(path):
    vsl = straight_line_velocity(path)
    vcl = culvilinear_velocity(path)
    return vsl / vcl


# WOB
def culvilinear_path_wobbling(path):
    vap = average_line_velocity(path)
    vcl = culvilinear_velocity(path)
    return vap / vcl


# STR
def average_path_straightness(path):
    vsl = straight_line_velocity(path)
    vcl = culvilinear_velocity(path)
    return vsl / vcl


# BCF - missing from paper equations
def average_path_crossing_colvilinear_path(path):
    avg_path = np.linspace(path[0], path[-1], path.shape[0])
    # turn into np.array
    avg_path = np.array(avg_path)
    diffs = path - avg_path
    cross_count = np.sum(diffs[1:] * diffs[:-1] < 0)
    bcf = cross_count / (path.shape[0] - 1)
    return bcf


# ALH - missing from paper equations
# I think we can't compute as no head data
def amplitude_of_lateral_head_displacement(path):
    pass


# MAD
def mean_angular_displacement(path):
    diffs = np.diff(path, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    mean_angle = np.mean(angles)
    return mean_angle
