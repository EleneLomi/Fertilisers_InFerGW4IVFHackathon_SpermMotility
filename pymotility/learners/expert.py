from preprocessing import *
import numpy as np
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
import numpy as np
import os


class MixtureOfExperts:
    def __init__(self, data):
        self.data = data
        self.preprocess()
        self.initialize_models()

    def preprocess(self):
        # Assuming segment_paths, recenter_paths, rotate_paths, and set_final_point_on_x_axis_path are defined
        data_processed = segment_paths(self.data)
        self.length_paths = len(data_processed[0])
        data_processed = recenter_paths(data_processed)
        data_rotated = rotate_paths(data_processed)
        data_final_point_on_x_axis = set_final_point_on_x_axis_path(data_rotated)
        self.data_rotated = np.vstack(
            [path[:, 0].reshape(-1, 1) for path in data_rotated]
        )
        self.data_final_point_on_x_axis = data_final_point_on_x_axis
        self.compute_distance_matrix(data_processed)
        self.variables = compute_paths_variables(data_processed)

    def compute_distance_matrix(self, data):
        n_samples = len(data)
        self.dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance_x = dtw.distance_fast(
                    data[i][:, 0], data[j][:, 0], use_pruning=True
                )
                distance_y = dtw.distance_fast(
                    data[i][:, 1], data[j][:, 1], use_pruning=True
                )
                self.dist_matrix[i, j] = self.dist_matrix[j, i] = (
                    distance_x + distance_y
                )

    def initialize_models(self):
        self.model_1 = KMedoids(n_clusters=3, random_state=0)
        self.model_2 = KMeans(n_clusters=3, random_state=0)
        self.model_3 = KMeans(n_clusters=3, random_state=0)

    def train(self):
        self.model_1.fit(self.dist_matrix)
        self.model_2.fit(self.variables)
        self.model_3.fit(self.data_rotated)

        self.make_labels_congruent()

    def make_labels_congruent(self):
        # Get the labels from each model
        labels_1 = self.model_1.labels_
        labels_2 = self.model_2.labels_
        labels_3 = self.model_3.labels_

        # Assuming all models are trained on the same dataset, we can try to align the labels
        # by finding the most common label mapping between them
        all_labels = np.vstack([labels_1, labels_2, labels_3])

        # Initialize new labels arrays to store the aligned labels
        new_labels_1 = np.empty_like(labels_1)
        new_labels_2 = np.empty_like(labels_2)
        new_labels_3 = np.empty_like(labels_3)

        for label in np.unique(labels_1):
            # Find the most common corresponding label in labels_2 and labels_3 for each label in labels_1
            idx = labels_1 == label
            most_common_label_2 = mode(labels_2[idx])[0][0]
            most_common_label_3 = mode(labels_3[idx])[0][0]

            # Assign the new labels
            new_labels_1[idx] = label
            new_labels_2[labels_2 == most_common_label_2] = label
            new_labels_3[labels_3 == most_common_label_3] = label

        # After alignment, the labels should be more consistent across models
        self.aligned_labels_1 = new_labels_1
        self.aligned_labels_2 = new_labels_2
        self.aligned_labels_3 = new_labels_3

    def plot_dendrogram(self):
        # Compute and plot the dendrogram
        linkage_matrix = linkage(squareform(self.dist_matrix), method="ward")
        dendrogram(linkage_matrix)
        plt.show()

    def plot_clusters(self):
        # Plot the clusters
        plt.scatter(
            self.data_final_point_on_x_axis[:, 0],
            self.data_final_point_on_x_axis[:, 1],
            c=self.aligned_labels_1,
        )
        plt.show()

    def predict_one_path(self, path):
        # Preprocess the input path
        path_processed = recenter_paths(path)
        path_rotated = rotate_paths(path_processed)
        path_final_point_on_x_axis = set_final_point_on_x_axis_path(path_rotated)

        # Compute the distance to the training data
        distance_x = dtw.distance_fast(
            path_final_point_on_x_axis[0][:, 0],
            self.data_final_point_on_x_axis[:, 0],
            use_pruning=True,
        )
        distance_y = dtw.distance_fast(
            path_final_point_on_x_axis[0][:, 1],
            self.data_final_point_on_x_axis[:, 1],
            use_pruning=True,
        )
        distance = distance_x + distance_y

        # Compute the variables
        variables = compute_paths_variables(path_processed)

        # Compute the rotated path
        path_rotated = np.vstack([path[:, 0].reshape(-1, 1)])

        # Predict the cluster
        label_1 = self.model_1.predict(distance.reshape(1, -1))[0]
        label_2 = self.model_2.predict(variables)[0]
        label_3 = self.model_3.predict(path_rotated)[0]

        # use congruent labels
        label_1 = self.aligned_labels_1[label_1]
        label_2 = self.aligned_labels_2[label_2]
        label_3 = self.aligned_labels_3[label_3]

        # Return the most common label
        return mode([label_1, label_2, label_3])[0][0]

    def predict(self, path):
        paths = segment_paths_to_given_length(path, self.length_paths)
        predictions = []
        for path in paths:
            predictions.append(self.predict_one_path(path))
        return mode(predictions)[0][0]

    def detect_anomaly_single(self, new_path):
        # Preprocess the new path
        new_path_processed = self.preprocess_path(new_path)

        # Initialize a counter for models considering the path as an anomaly
        anomaly_count = 0

        medoid_distances = []
        for memoid in self.model_1.cluster_centers_:
            distance_x = dtw.distance_fast(
                new_path_processed[0][:, 0], memoid[:, 0], use_pruning=True
            )
            distance_y = dtw.distance_fast(
                new_path_processed[0][:, 1], memoid[:, 1], use_pruning=True
            )
            medoid_distances.append(distance_x + distance_y)

        distance_1 = np.min(medoid_distances)
        distances_2 = np.min(
            pairwise_distances([self.variables], self.model_2.cluster_centers_), axis=1
        )
        distances_3 = np.min(
            pairwise_distances([new_path_processed], self.model_3.cluster_centers_),
            axis=1,
        )

        # Define your threshold for considering a path too far from clusters
        threshold = 1  # Define based on your data and model characteristics

        # Check distances against threshold
        if distance_1 > threshold:
            anomaly_count += 1
        if distances_2 > threshold:
            anomaly_count += 1
        if distances_3 > threshold:
            anomaly_count += 1

        # Check if the path is considered an anomaly by at least 2 out of 3 models
        if anomaly_count >= 2:
            return True
        else:
            return False

    def detect_anomaly(self, new_path):
        new_paths = segment_paths_to_given_length(new_path, self.length_paths)
        anomalies = []
        for path in new_paths:
            anomalies.append(self.detect_anomaly_single(path))
        return mode(anomalies)[0][0]


if __name__ == "__main__":
    #  load all the files in "../../data/sample_1_paths/"
    #  and store them in the list "data"
    data = []
    # for each file in the directory
    for filename in os.listdir("../../data/path_extraction/sample_1_paths/dof/"):
        # load the file
        path = np.load("../../data/path_extraction/sample_1_paths/dof/" + filename)
        # append the path to the data list
        data.append(path)

    expert = MixtureOfExperts(data)
