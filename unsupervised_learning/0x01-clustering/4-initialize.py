#!/usr/bin/env python3
""" Initialize """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ Initialize variables for a Gaussian Mixture Model"""
    try:
        _, d = X.shape
        centroids, _ = kmeans(X, k)
        pi = np.ones((k)) / k
        S = np.array([np.identity(d)] * k)
        return pi, centroids, S
    except Exception:
        return None, None, None
