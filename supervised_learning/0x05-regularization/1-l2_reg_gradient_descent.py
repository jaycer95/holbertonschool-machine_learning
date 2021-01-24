#!/usr/bin/env python3
""" Regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0 , -1):
        dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
        db = np.sum(dz, axis = 1, keepdims=True) / m
        dA = 1 - cache["A" + str(i - 1)] * cache["A" + str(i - 1)]
        dz = np.matmul(weights["W" + str(i)].T, dz) * dA

        weights["W" + str(i)] = weights["W" + str(i)] - alpha * (dw + weights["W" + str(i)]) * (lambtha / m)
        weights["b" + str(i)] = weights["W" + str(i)] - alpha * db