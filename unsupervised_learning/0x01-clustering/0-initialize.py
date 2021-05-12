#!/usr/bin/env python
""" Clustering """

import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-mean"""
    try:
        _, d = X.shape
        centroids = np.random.uniform(
            low=np.min(
                X, axis=0), high=np.max(
                X, axis=0), size=(
                k, d))
    except Exception:
        return None
    return centroids
