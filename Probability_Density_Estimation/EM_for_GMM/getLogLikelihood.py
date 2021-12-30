import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Start Subtask 6a#####
    if len(X.shape) > 1:
        N, D = X.shape
    else:
        N = 1
        D = X.shape[0]

    # get number of gaussians
    K = len(weights)

    logLikelihood = 0
    for i in range(N):  # For each of the data points
        # probability p
        p = 0
        for j in range(K):  # For each of the mixture components

            if N == 1:
                meansDiff = X - means[j]
            else:
                meansDiff = X[i,:] - means[j]

            covariance = covariances[:, :, j].copy()
            norm = 1. / float(((2 * np.pi) ** (float(D) / 2.)) * np.sqrt(np.linalg.det(covariance)))


            p += weights[j] * norm * np.exp(-0.5 * ((meansDiff.T).dot(np.linalg.lstsq(covariance.T, meansDiff.T)[0].T)))
        logLikelihood += np.log(p)
    #####End Subtask#####
    return logLikelihood

