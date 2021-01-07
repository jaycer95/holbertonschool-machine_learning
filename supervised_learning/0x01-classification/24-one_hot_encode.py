#!/usr/bin/env python3

import numpy as np


def one_hot_encode(Y, classes):
    """ One hot encode """
    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
