import numpy as np
import sys
from conversions_and_misc import *

def hoshen_kopelman(matrix):
    """
    Hoshen-Kopelman algorithm for cluster labeling in a 3D lattice.

    Args:
    matrix (np.array): 3D numpy array representing the binary lattice.

    Returns:
    np.array: Labeled matrix where each cluster has a unique integer label.
    """
    sys.setrecursionlimit(100000)

    m, n, l = matrix.shape
    labels = np.zeros((m, n, l), dtype=int)
    label = 1

    labelled = np.zeros((m, n, l), dtype=int)
    particle_label = 1
    cluster_points = {}
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if matrix[i, j, k] == 1 and labelled[i, j, k] == 0:

                    particle_label+=1
                    label_neighbours(i, j, k, matrix, particle_label, labelled, labels, cluster_points)


    return labels

def label_neighbours(i, j, k, matrix, particle_label, labelled, labels, cluster_points):
    #Abbruch-Bedingung
    if matrix[i, j, k] == 0 or labelled[i, j, k] == 1:
        return
    labelled[i, j, k] = 1
    labels[i, j, k] = particle_label

    #rekursiver Aufruf
    label_neighbours((i + 1) % matrix.shape[0], j, k, matrix, particle_label, labelled, labels, cluster_points)
    label_neighbours((i - 1) % matrix.shape[0], j, k,  matrix, particle_label, labelled, labels, cluster_points)
    label_neighbours(i, (j + 1) % matrix.shape[1], k, matrix, particle_label, labelled, labels, cluster_points)
    label_neighbours(i, (j - 1) % matrix.shape[1], k, matrix, particle_label, labelled, labels, cluster_points)
    label_neighbours(i, j, (k + 1) % matrix.shape[2], matrix, particle_label, labelled, labels, cluster_points)
    label_neighbours(i, j, (k - 1) % matrix.shape[2], matrix, particle_label, labelled, labels, cluster_points)



def hk_randomize_labels(hk_matrix):
    res_matrix = np.zeros(hk_matrix.shape, dtype=hk_matrix.dtype)
    cluster_count = np.max(hk_matrix)
    shuffled_labels = np.random.permutation(cluster_count) + 1
    print(shuffled_labels)

    for label in range(1, cluster_count + 1):
        res_matrix[hk_matrix == label] = shuffled_labels[label - 1]

    return res_matrix


if __name__ == '__main__':

    path2npy = 'output/20240815_170409xgb_array_output_64_window_size13_3D/npy/8.npy'

    ms = np.load(path2npy)

    ms_k = hoshen_kopelman(ms)
    shuffled_labels_hk = hk_randomize_labels(ms_k)
    convert_np_to_vti(shuffled_labels_hk,f'./output.vti')

