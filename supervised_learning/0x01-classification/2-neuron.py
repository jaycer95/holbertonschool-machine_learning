#!/usr/bin/env python3
""" Neuron Class """


import numpy as np


class Neuron:
    """ Define a single neuron performing binary classification """

    def __init__(self, nx):
        """ Instantination """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """

        z = np.dot(self.__W, X) + self.__b  # Preactivation parameter
        self.__A = 1 / (1 + np.exp(- z))
        return self.__A
