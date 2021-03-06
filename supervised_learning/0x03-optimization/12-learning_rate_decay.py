#!/usr/bin/env python3
""" Normalization function """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """  creates the training operation for a neural network   """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
