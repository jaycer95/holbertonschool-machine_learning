#!/usr/bin/env python3
""" Forward propagation """

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network """
    l = x
    for i in range(len(layer_sizes)):
        l = create_layer(l, layer_sizes[i],
                         activation=activations[i])
    return l
