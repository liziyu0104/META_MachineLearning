import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    #####Start Subtask 6e#####
    n_dim = data.shape[1]

    # initialize weights and covariances
    weights = np.ones(K) / K
    covariances = np.zeros((n_dim, n_dim, K))

    # Use k-means for initializing the EM-Algorithm.
    # cluster_idx: cluster indices
    # means: cluster centers
    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(n_dim) * min_dist

    # Run EM
    for idx in range(n_iters):

        oldLogLi, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, newLogli = MStep(gamma, data)

        # regularize covariance matrix
        for j in range(K):
            covariances[:, :, j] = regularize_cov(covariances[:, :, j], epsilon)

        # termination criterion
        if abs(oldLogLi - newLogli) < 1:
            break

    #####End Subtask#####
    return [weights, means, covariances]
