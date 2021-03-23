#!/usr/bin/env python3
""" Dimensionality Reduction"""

import numpy as np
from numpy import cov
from numpy import mean
from numpy.linalg import eig

def pca(X, var=0.95):
    """ Principal component analysis """
    cov = np.cov(X.T) / X.shape[0]
    eigen_values , eigen_vectors = np.linalg.eig(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    return -sorted_eigenvectors[:,:3]