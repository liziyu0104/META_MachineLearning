import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Start Subtask 5b#####
    # Compute the number of the samples created
    N = len(samples)

    # Create a linearly spaced vector
    pos = np.arange(-5, 5.0, 0.1)

    # Sort the distances so that we can choose the k-th point
    dists = np.sort(np.abs(pos[np.newaxis, :] - samples[:, np.newaxis]), axis=0)

    # Estimate the probability density using the k-NN density estimation
    res = (k / (2 * N)) / dists[k - 1, :]

    # Form the output variable
    estDensity = np.stack((pos, res), axis=1)

    #####End Subtask#####
    return estDensity
