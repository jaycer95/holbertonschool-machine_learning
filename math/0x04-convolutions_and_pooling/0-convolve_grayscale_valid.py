#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ perform a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = (images[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
