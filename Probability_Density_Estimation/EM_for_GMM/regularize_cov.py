import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Start Subtask 6d#####
    n, m = covariance.shape
    regularized_cov = covariance + epsilon * np.eye(n, m)

    # makes sure matrix is symmetric upto 1e-15 decimal
    regularized_cov = (regularized_cov + regularized_cov.conj().transpose()) / 2

    #####End Subtask#####
    return regularized_cov
