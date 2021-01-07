#!/usr/bin/env python3
""" Create layers """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    logits = tf.transpose(y_pred)
    labels = tf.transpose(y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))