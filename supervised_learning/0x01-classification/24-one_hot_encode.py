#!/usr/bin/env python3

import numpy as np


def one_hot_encode(Y, classes):
    """ One hot encode """
    if not isinstance(Y, np.ndarray):
        return None
    try:
        return np.eye(classes)[Y.reshape(-1)].T
    except Exception:
        return None
