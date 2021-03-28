#!/usr/bin/env python3
""" Dimensionality Reduction"""

import numpy as np


def pca(X, ndim):
    """ Principal component analysis """
    Xcentered = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(Xcentered)
    v = np.cumsum(S) / np.sum(S)
    W = V.T[:, :ndim]
    T = np.matmul(Xcentered, W)
    return T