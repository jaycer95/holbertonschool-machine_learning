#!/usr/bin/env python3
""" Keras """

from tensorflow import keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        verbose=True,
        shuffle=False):
    """ train a model using mini-batch gradient descent """
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle)
