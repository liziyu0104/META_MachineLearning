import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Start Subtask 6b#####
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    n_training_samples, dim = X.shape
    K = len(weights)

    gamma = np.zeros((n_training_samples, K))
    for i in range(n_training_samples):
        for j in range(K):
            means_diff = X[i] - means[j]
            covariance = covariances[:, :, j].copy()
            norm = 1. / float(((2 * np.pi) ** (float(dim) / 2)) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            gamma[i, j] = weights[j] * norm * np.exp(
                -0.5 * (means_diff.T.dot(np.linalg.lstsq(covariance.T, means_diff.T)[0].T)))
        gamma[i] /= gamma[i].sum()

    #####End Subtask#####
    return [logLikelihood, gamma]
