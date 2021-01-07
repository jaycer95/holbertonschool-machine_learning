#!/usr/bin/env python3
""" One hot decode """
import numpy as np


def one_hot_decode(one_hot):
    """ One hot decode """
    try:
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
