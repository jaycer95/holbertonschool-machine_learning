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

        P = self.__A.astype(int)
        P[P >= 0.5] = 1
        P[P < 0.5] = 0
        return P, self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """

        m = A.shape[1]
        dw = np.dot((A - Y), X.T) / m
        db = (np.sum(A - Y)) / m

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train the neuron """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
