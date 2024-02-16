import numpy as np
from motility import *


def segment_paths(paths, overlap=1):
    # Calculate lengths of all paths
    lengths = [len(path) for path in paths]
    min_length = min(lengths)  # Find the shortest path length

    segmented_paths = []
    for path in paths:
        if len(path) > min_length:
            for i in range(0, len(path), min_length - overlap):
                # Split paths longer than the shortest one
                segmented_path = path[i : i + min_length]
                if len(segmented_path) == min_length:
                    segmented_paths.append(segmented_path)
        else:
            segmented_paths.append(path)  # Add the shortest paths as is

    return segmented_paths


def segment_paths_to_given_length(paths, length):
    segmented_paths = []
    for path in paths:
        if len(path) > length:
            for i in range(0, len(path), length):
                # Split paths longer than the given length
                segmented_path = path[i : i + length]
                if len(segmented_path) == length:
                    segmented_paths.append(segmented_path)
        else:
            segmented_paths.append(path)  # Add the shortest paths as is

    return segmented_paths


def recenter_paths(paths):
    renormalized_paths = []
    for path in paths:
        # Subtract the first point from all points in the path
        origin_point = path[0, :]
        renormalized_path = path - origin_point[None, :]
        renormalized_paths.append(renormalized_path)
    return renormalized_paths


def rotate_path(path):
    # Ensure path is in the shape [2, T] for the operations
    path_transposed = path.T

    # Calculate the mean of each dimension
    mean = path_transposed.mean(axis=1, keepdims=True)

    # Center the path around the origin
    centered_path = path_transposed - mean

    # Calculate the covariance matrix of the centered path
    covariance_matrix = np.cov(centered_path)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Principal component is the eigenvector with the largest eigenvalue
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate the angle to the x-axis
    angle_to_x_axis = np.arctan2(principal_component[1], principal_component[0])

    # Rotation matrix to align the principal component with the x-axis
    rotation_matrix = np.array(
        [
            [np.cos(-angle_to_x_axis), -np.sin(-angle_to_x_axis)],
            [np.sin(-angle_to_x_axis), np.cos(-angle_to_x_axis)],
        ]
    )

    # Rotate the path
    rotated_path = np.dot(rotation_matrix, centered_path)

    return rotated_path.T  # Transpose back to original shape [T, 2]


def rotate_paths(paths):
    rotated_paths = []
    for path in paths:
        rotated_paths.append(rotate_path(path))
    return rotated_paths


def set_final_point_on_x_axis_path(path):
    # Calculate the angle between the last point and the x-axis
    final_point = path[-1]
    angle = -np.arctan2(final_point[1], final_point[0])

    # Rotation matrix
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Rotate all points
    rotated_path = np.dot(path, rotation_matrix.T)
    return rotated_path


def set_final_point_on_x_axis(paths):
    rotated_paths = []
    for path in paths:
        rotated_paths.append(set_final_point_on_x_axis_path(path))
    return rotated_paths


def compute_paths_variables(paths):
    paths_variables = []
    for path in paths:
        paths_variables.append(compute_path_variables(path))
    return paths_variables


def compute_path_variables(path):
    vcl = culvilinear_velocity(path)
    vsl = straight_line_velocity(path)
    vap = average_line_velocity(path)
    lin = linearity_progressive_motility(path)
    wob = culvilinear_path_wobbling(path)
    str_a = average_path_straightness(path)
    bcf = average_path_crossing_colvilinear_path(path)
    mad = mean_angular_displacement(path)
    return [vcl, vsl, vap, lin, wob, str_a, bcf, mad]
