#!/usr/bin/env python3
""" DeepNeuralNetwork Class """


import numpy as np


class DeepNeuralNetwork:
    """ Define a Deep Neural Network performing binary classification """

    def __init__(self, nx, layers):
        """ Instantination """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not (layers > 0).all() and not all(
                map(lambda x: isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")
        self.L = 0
        self.cache = {}
        self.weights = {}
        leng = len(layers) - 1  # integer representing the number of layers
        for l in range(1, leng + 1):
            self.weights["W" + str(l)] = np.random.randn(layers[l],
                                                         nx) * np.sqrt(2 / nx)
            self.weights["b" + str(l)] = np.zeros((layers[l], 1))
