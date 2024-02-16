from preprocessing import *
import numpy as np
from tslearn.metrics import dtw
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
import numpy as np
import os
import matplotlib.pyplot as plt

class MixtureOfExperts:
    def __init__(self, data, debug = False):
        self.data = data
        self.debug = debug
        self.preprocess(debug)
        self.initialize_models()

    def preprocess(self, debug):
        # Assuming segment_paths, recenter_paths, rotate_paths, and set_final_point_on_x_axis_path are defined
        data_processed = segment_paths(self.data)
        self.length_paths = len(data_processed[0])
        data_processed = recenter_paths(data_processed)
        data_rotated = rotate_paths(data_processed)
        data_final_point_on_x_axis = set_final_point_on_x_axis(data_rotated)
        self.data_rotated = np.vstack(
            [path[:, 0] for path in data_rotated]  # Selects only the first coordinate, keeping the shape as (T, 1)
        )
        self.data_final_point_on_x_axis = data_final_point_on_x_axis
        self.compute_distance_matrix(data_processed)
        variables = compute_paths_variables(data_processed)
        variables = [np.array(path) for path in variables]
        self.variables = variables

    def compute_distance_matrix(self, data):
        n_samples = len(data)
        self.dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance_x = dtw(
                    data[i][:, 0], data[j][:, 0], global_constraint="sakoe_chiba", sakoe_chiba_radius=3
                )
                distance_y = dtw(
                    data[i][:, 1], data[j][:, 1], global_constraint="sakoe_chiba", sakoe_chiba_radius=3
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

        # print the size of all clusters
        print("Size of clusters in model 1: ", np.bincount(self.model_1.labels_))
        print("Size of clusters in model 2: ", np.bincount(self.model_2.labels_))
        print("Size of clusters in model 3: ", np.bincount(self.model_3.labels_))

        self.make_labels_congruent(self.debug)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        # Convert y_true to one-hot encoded vectors if not already
        n_classes = y_pred.shape[1]
        y_true_one_hot = np.eye(n_classes)[y_true]

        # Clip y_pred to prevent numerical issues with log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate cross-entropy
        ce = -np.sum(y_true_one_hot * np.log(y_pred)) / y_true.shape[0]
        return ce

    def make_labels_congruent(self, debug = False):
        pass

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
        distance_x = dtw(
            path_final_point_on_x_axis[0][:, 0],
            self.data_final_point_on_x_axis[:, 0],
            global_constraint="sakoe_chiba", sakoe_chiba_radius=3
        )
        distance_y = dtw(
            path_final_point_on_x_axis[0][:, 1],
            self.data_final_point_on_x_axis[:, 1],
            global_constraint="sakoe_chiba", sakoe_chiba_radius=3,
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
        return mode([label_1, label_2, label_3])[0]

    def predict(self, path):
        paths = segment_paths_to_given_length(path, self.length_paths)
        predictions = []
        for path in paths:
            predictions.append(self.predict_one_path(path))
        return mode(predictions)[0]

    def detect_anomaly_single(self, new_path):
        # Preprocess the new path
        new_path_processed = self.preprocess_path(new_path)

        # Initialize a counter for models considering the path as an anomaly
        anomaly_count = 0

        medoid_distances = []
        for memoid in self.model_1.cluster_centers_:
            distance_x = dtw(
                new_path_processed[0][:, 0], memoid[:, 0], global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
            distance_y = dtw(
                new_path_processed[0][:, 1], memoid[:, 1], global_constraint="sakoe_chiba", sakoe_chiba_radius=3
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
        return mode(anomalies)[0]
    
    def info(self):
        # print the centers of the clusters of model 2
        centers = self.model_2.cluster_centers_
    # vcl = culvilinear_velocity(path)
    # vsl = straight_line_velocity(path)
    # vap = average_line_velocity(path)
    # lin = linearity_progressive_motility(path)
    # wob = culvilinear_path_wobbling(path)
    # str_a = average_path_straightness(path)
    # bcf = average_path_crossing_colvilinear_path(path)
    # mad = mean_angular_displacement(path)
        for i in range(len(centers)):
            print("Center Index Congruent Label: ", self.aligned_labels_2[i])
            print("vcl: ", centers[i][0])
            print("vsl: ", centers[i][1])

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

    expert = MixtureOfExperts(data, debug = True)
    expert.train()
    expert.info()

