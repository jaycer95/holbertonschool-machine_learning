#!/usr/bin/env python3
"""K-means"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        n, d = X.shape
        low_b = np.min(X, 0)
        high_b = np.max(X, 0)
        centroids = np.random.uniform(low_b, high_b, (k, d))
    except Exception:
        return None
    return centroids


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    centroids = initialize(X, k)
    n, d = X.shape
    if centroids is None:
        return None, None
    for _ in range(iterations):
        centroids_copy = np.copy(centroids)
        dist = np.linalg.norm((X[:, np.newaxis, :] - centroids),axis=2)
        clusters = dist.argmin(1)
        for i in range(k):
            if i in clusters:
                centroids[i,:] = X[clusters == i, :].mean(0)
            else:
                k == 1
                centroids = initialize(X, k)
        if (centroids_copy == centroids).all():
            break
        else:
            centroids_copy = np.copy(centroids)
    return centroids, clusters
