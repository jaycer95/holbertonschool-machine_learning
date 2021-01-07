#!/usr/bin/env python3
""" Create layers """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    a = tf.metrics.accuracy(y, y_pred)
    accuracy = tf.math.reduce_mean(tf.cast(a, tf.float32))
    return accuracy


    # logits = tf.transpose(y_pred)
    # labels = tf.transpose(y)
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))