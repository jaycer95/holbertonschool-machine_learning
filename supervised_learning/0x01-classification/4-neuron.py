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

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """

        m = Y.shape[1]
        T = np.transpose(A)
        L = np.transpose(np.log(1.0000001 - A))
        cst = np.squeeze(-1 / m * (np.dot(Y, np.log(T)) + np.dot(1 - Y, L)))
        return cst

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """

        # a = self.forward_prop(X)
        # P = a.astype(int)
        # P[P >= 0.5] = 1
        # P[P < 0.5] = 0
        m = X.shape[1]
        P = np.zeros((1, m), dtype=int)
        for i in range(a.shape[1]):
            if a[0, i] >= 0.5:
                P[0, i] = 1
            else:
                P[0, i] = 0
        c = self.cost(Y, a)
        return P, c
