import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Start Subtask 6c#####
    
    # Get the sizes
    n_training_samples, dim = X.shape
    K = gamma.shape[1]

    # Create matrices
    means = np.zeros((K, dim))
    covariances = np.zeros((dim, dim, K))

    # Compute the weights
    Nk = gamma.sum(axis=0)
    weights = Nk / n_training_samples

    # for i in range(K):
    #     auxMean = np.zeros(dim)
    #     for j in range(n_training_samples):
    #         auxMean += gamma[j, i] * X[j]
    #     means[i] = auxMean / Nk[i]
    means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])
    # assert means_new.all() == means.all()

    for i in range(K):
        auxSigma = np.zeros((dim, dim))
        for j in range(n_training_samples):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma/Nk[i]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    #####End Subtask#####
    return weights, means, covariances, logLikelihood
