#!/usr/bin/env python3
""" Keras """

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ save a model’s weights """
    network.save_weights(filename, save_format='h5')
    return None


def load_weights(network, filename):
    """load a model’s weights"""
    network.load_weights(filename)
    return None
