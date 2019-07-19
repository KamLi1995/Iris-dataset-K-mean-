import numpy as np
import scipy as sp
import csv
import pandas as pd
import random
import math
from decimal import *
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


def k_init(X, k):

    total_centroids = np.zeros((k,2),dtype=X.dtype)
    first_centroids = X[random.randint(0,X.shape[0])]

    total_centroids[0] = first_centroids
    repeat_centroids = np.zeros(shape = X.shape, dtype = X.dtype)

    ###############################################################
    ##this section is used for calculating the euclidean distance##
    for i in range(X.shape[0]):
        repeat_centroids[i] = first_centroids

    tempholdersubtract = np.zeros(shape = X.shape, dtype = X.dtype)
    tempholdersquare = np.zeros(shape = X.shape, dtype = X.dtype)
    tempholderadd = np.zeros(shape = X.shape[0])
    tempholdersqrt = np.zeros(shape = X.shape[0])

    for i in range(X.shape[0]):
        tempholdersubtract[i][0] = abs(X[i][0] - repeat_centroids[i][0])
        tempholdersubtract[i][1] = abs(X[i][1] - repeat_centroids[i][1])

    for i in range(X.shape[0]):
        tempholdersquare[i][0] = (tempholdersubtract[i][0])**2
        tempholdersquare[i][1] = (tempholdersubtract[i][1])**2

    for i in range(X.shape[0]):
        tempholderadd[i] = tempholdersquare[i][0] + tempholdersquare[i][1]

    for i in range(X.shape[0]):
        tempholdersqrt[i] = math.sqrt(tempholderadd[i])
    ##################################################################

    #######################################################################################################
    ## if our number of cluster is greater than 1 we will find next centroids with a weighted probability##
    if k >= 2:
        for j in range(1,k):
            probs = tempholdersqrt/tempholdersqrt.sum()
            total_centroids[j] = X[np.random.choice(np.arange(probs.shape[0]), p=probs)]

            temp_centroids = np.zeros(shape = X.shape, dtype = X.dtype)

            for i in range(X.shape[0]):
                temp_centroids[i] = total_centroids[j]

            for i in range(X.shape[0]):
                tempholdersubtract[i][0] = abs(X[i][0] - temp_centroids[i][0])
                tempholdersubtract[i][1] = abs(X[i][1] - temp_centroids[i][1])

            for i in range(X.shape[0]):
                tempholdersquare[i][0] = (tempholdersubtract[i][0]) ** 2
                tempholdersquare[i][1] = (tempholdersubtract[i][1]) ** 2

            for i in range(X.shape[0]):
                tempholderadd[i] = tempholdersquare[i][0] + tempholdersquare[i][1]

            for i in range(X.shape[0]):
                tempholdersqrt[i] = math.sqrt(tempholderadd[i])
    #######################################################################################################
    return total_centroids

def k_means_pp(X, k, max_iter):
    centroids = k_init(X,k)
    euclidean_distance = np.zeros(shape=(X.shape[0],k), dtype=X.dtype)
    temp_centroids = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersubtract = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersquare = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholderadd = np.zeros(shape=X.shape[0])
    tempholdersqrt = np.zeros(shape=X.shape[0])

    #######################################################################################################
    ## finding new centroid after every iteration
    for i in range(max_iter):
        for i in range(k):
            for j in range(X.shape[0]):
                temp_centroids[j] = centroids[i]
            for j in range(X.shape[0]):
                tempholdersubtract[j][0] = abs(X[j][0] - temp_centroids[j][0])
                tempholdersubtract[j][1] = abs(X[j][1] - temp_centroids[j][1])

            for j in range(X.shape[0]):
                tempholdersquare[j][0] = (tempholdersubtract[j][0]) ** 2
                tempholdersquare[j][1] = (tempholdersubtract[j][1]) ** 2

            for j in range(X.shape[0]):
                tempholderadd[j] = tempholdersquare[j][0] + tempholdersquare[j][1]

            for j in range(X.shape[0]):
                tempholdersqrt[j] = math.sqrt(tempholderadd[j])
                euclidean_distance[j][i] = tempholdersqrt[j]
    #######################################################################################################

        closest_centroids = (np.argmin((euclidean_distance),axis = 1))

        Lists = [[] for i in range(k)]
        for i in range(0, X.shape[0]):
            n = np.asscalar(closest_centroids[i])
            Lists[n].append(X[i, :])

        calculated_centroid = np.ndarray(shape=(0, centroids.shape[1]))

        for i in range(0, k):
            cluster_assignment = np.asmatrix(Lists[i])
            centroidCluster = cluster_assignment.mean(axis=0)
            calculated_centroid = np.concatenate((calculated_centroid, centroidCluster), axis=0)

        centroids = calculated_centroid

    return centroids




def assign_data2clusters(X, C):

    centroids = C
    k = C.shape[0]
    euclidean_distance = np.zeros(shape=(X.shape[0], k), dtype=X.dtype)
    temp_centroids = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersubtract = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersquare = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholderadd = np.zeros(shape=X.shape[0])
    tempholdersqrt = np.zeros(shape=X.shape[0])

    for i in range(k):
        for j in range(X.shape[0]):
            temp_centroids[j] = centroids[i]
        for j in range(X.shape[0]):
            tempholdersubtract[j][0] = abs(X[j][0] - temp_centroids[j][0])
            tempholdersubtract[j][1] = abs(X[j][1] - temp_centroids[j][1])

        for j in range(X.shape[0]):
            tempholdersquare[j][0] = (tempholdersubtract[j][0]) ** 2
            tempholdersquare[j][1] = (tempholdersubtract[j][1]) ** 2

        for j in range(X.shape[0]):
            tempholderadd[j] = tempholdersquare[j][0] + tempholdersquare[j][1]

        for j in range(X.shape[0]):
            tempholdersqrt[j] = math.sqrt(tempholderadd[j])
            euclidean_distance[j][i] = tempholdersqrt[j]


        closest_centroids = (np.argmin((euclidean_distance),axis = 1))

        #######################################################################################################
        ## mapping out data to the nearest centroid

        Lists = [[] for i in range(k)]
        for i in range(0, X.shape[0]):
            n = np.asscalar(closest_centroids[i])
            Lists[n].append(X[i, :])

    return Lists


def compute_objective(X, C):
    objective = 0.0
    data_map = assign_data2clusters(X, C)
    final_centroids = C

    euclidean_distance = np.zeros(shape=(X.shape[0], C.shape[0]), dtype=X.dtype)
    temp_centroids = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersubtract = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholdersquare = np.zeros(shape=X.shape, dtype=X.dtype)
    tempholderadd = np.zeros(shape=X.shape[0])
    tempholdersqrt = np.zeros(shape=X.shape[0])
    objectivematrix = np.zeros(shape=X.shape, dtype=X.dtype)

    #######################################################################################################
    ## calculating the objective by adding up the distances

    for k in range(len(data_map)):
        n = len(data_map[k])

        for i in range(n):
            for j in range(X.shape[0]):
                temp_centroids[j] = final_centroids[k]

            tempholdersubtract[i][0] = abs(X[i][0] - temp_centroids[0][0])
            tempholdersubtract[i][1] = abs(X[i][1] - temp_centroids[0][1])

            tempholdersquare[i][0] = (tempholdersubtract[i][0]) ** 2
            tempholdersquare[i][1] = (tempholdersubtract[i][1]) ** 2

            tempholderadd[i] = tempholdersquare[i][0] + tempholdersquare[i][1]

            tempholdersqrt[i] = math.sqrt(tempholderadd[i])
            objective = objective + tempholdersqrt[i]
    #######################################################################################################


    return objective


if __name__ == "__main__":

    with open('iris.csv') as data:
        iris_data_set = csv.reader(data, delimiter=',')

        next(iris_data_set)

        S_lengths = []
        S_widths = []
        P_lengths = []
        P_widths = []
        Species = []

        for row in iris_data_set:
            S_length = row[0]
            S_width = row[1]
            P_length = row[2]
            P_width = row[3]
            Specie = row[4]

            S_lengths.append(S_length)
            S_widths.append(S_width)
            P_lengths.append(P_length)
            P_widths.append(P_width)
            Species.append(Specie)

    x1 = np.array([])
    x2 = np.array([])

    for i in range(len(S_lengths)):
        temp1 = float(Decimal(S_lengths[i]) / Decimal(S_widths[i]))
        temp2 = float(Decimal(P_lengths[i]) / Decimal(P_widths[i]))

        x1 = np.append(x1, temp1)
        x2 = np.append(x2, temp2)

    X = np.array([x1, x2])
    X = X.transpose()

    max_iter = 50
    number_of_cluster = 2

    centroid = k_means_pp(X,number_of_cluster,50)
    data_map = assign_data2clusters(X, centroid)

    number_of_iteration = list(range(1, max_iter + 1))
    accuracy = []
    for i in number_of_iteration:
        final_center = k_means_pp(X, number_of_cluster, i)
        accuracy.append(compute_objective(X, final_center))

    plt.title('Objective Function')
    plt.xlabel('i')
    plt.ylabel('Accuracy')
    plt.plot(number_of_iteration, accuracy)
    plt.show()


    n = len(data_map)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.clf()
    for i in range(n):
        next_color = next(color)
        cluster_assignment = np.asmatrix(data_map[i])
        plt.scatter(np.ravel(cluster_assignment[:, 0]), np.ravel(cluster_assignment[:, 1]), marker=".", s=100, c=next_color)
        plt.scatter((centroid[i, 0]), (centroid[i, 1]), marker="X", s=400, c=next_color, edgecolors="black")

    plt.pause(100)

