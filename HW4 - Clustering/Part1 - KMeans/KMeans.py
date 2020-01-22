import numpy as np
import pandas as pd
import random
import math
from math import sqrt
import sys
from Blackbox41 import Blackbox41
from Blackbox42 import Blackbox42
import csv


#Extracting Train data from Command Line Arguments
no_of_cli_arguments = len(sys.argv)
if no_of_cli_arguments!=2:
    print("Blackbox path not specified; Please check the CLI arguments")
    sys.exit(0)
args = sys.argv
train_df_path = args[1]
indice = train_df_path.index("blackbox")
output_name = "results_blackbox4{}.csv".format(train_df_path[indice+9])  #OutputFile Name


# Input Parameters

# Intializing blackbox
if train_df_path=="blackbox41":
    blackbox = Blackbox41()
elif train_df_path=="blackbox42":
    blackbox = Blackbox42()
else:
    print("invalid blackbox")
    sys.exit()

# Loading Data
train_data = blackbox.ask()

k=4  # Number of clusters
tolerance= 0.001
max_iterations = 200
random.seed(10)
clusters = {0:[],1:[],2:[],3:[]}
good_clusters_found = False
centroids= []

# Choosing k random centroids
random_k_indices = random.sample(range(0, train_data.shape[0]), k)
for i in range(k):
    centroids.append(train_data[i])


# Core algorithm for K-Means Clustering
def k_means(max_iterations):
    for count in range(max_iterations):
        clusters = {0:[],1:[],2:[],3:[]}
        good_clusters_found = True

        # For each training data point, find out the distance to nearest centroid and assign it to that cluster
        for row in train_data:
            distance = float("inf")
            for index, centroid in enumerate(centroids):
                curr_dist = np.linalg.norm(row-centroid)
                if curr_dist < distance:
                    distance = curr_dist
                    min_distance_index = index
            clusters[min_distance_index].append(row)

        # Stroing previous centroids before computing average on new clusters formed
        old_centroids = list(centroids)

        # New centroids are formed by averaging data points in each cluster
        for cluster in clusters:
            centroids[cluster] = np.average(clusters[cluster],axis=0)

        # Stop if new centroid is equal to old centroid
        for centroid in range(len(centroids)):
            if not all(centroids[centroid]==old_centroids[centroid]):
                good_clusters_found = False
                break
            # if np.sum((centroids[centroid] - old_centroids[centroid]) / old_centroids[centroid] * 100) > tolerance:
            #     good_clusters_found = False
            #     break
        if good_clusters_found: breakd

k_means(max_iterations)


# Assigning data ponits to clusters based on nearest distnace to k centroids
results = []
for row in train_data:
        distances = []
        for centroid in centroids:
            distances.append(np.linalg.norm(row-centroid))
        min_distance_index = distances.index(min(distances))
        results.append(min_distance_index)


#Writing Ouput i.e. predicted lables to a csv file
with open(output_name, 'w',newline='') as f:
    writer = csv.writer(f)
    for val in results:
        writer.writerow([val])
