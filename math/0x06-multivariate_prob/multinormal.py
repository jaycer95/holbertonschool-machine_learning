#!/usr/bin/env python3
"""Multivariate probabability"""

import numpy as np


class MultiNormal:
    """the class pf multinormal"""
    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        arg_mean = data - self.mean
        self.cov = (np.dot(arg_mean, arg_mean.T)) / (n - 1)
