#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """perform pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    stride_h, stride_w = stride

    output_h = int(np.floor((h - kh) / stride_h) + 1)
    output_w = int(np.floor((w - kw) / stride_w) + 1)

    output = np.zeros((m, output_h, output_w, c))

    for x in range(output_h):
        for y in range(output_w):
            if mode == "max":
                output[:, x, y, :] = np.max(
                    images[:, x*stride_h:kh+x*stride_h, y*stride_w:kw+y*stride_w, :],
                    axis=(1, 2))
            if mode == "avg":
                output[:, x, y, :] = np.average(
                    images[:, x*stride_h:kh+x*stride_h, y*stride_w:kw+y*stride_w, :],
                    axis=(1, 2))
    return output