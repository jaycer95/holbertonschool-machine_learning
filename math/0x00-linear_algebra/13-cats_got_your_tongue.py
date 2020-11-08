#!/usr/bin/env python3
"""Function for matrix operations"""
import numpy as np

def np_cat(mat1, mat2, axis=0):
    arr = np.concatenate((mat1, mat2), axis=axis)
    return arr
