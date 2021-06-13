#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


class BidirectionalCell:
    """ Represent a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ initialization"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax Activation Function"""
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid Activation Function"""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, x_t):
        """Calculate the hidden state in the forward
        direction for one time step """
        xh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(xh, self.Whf) + self.bhf)
        return h_next
