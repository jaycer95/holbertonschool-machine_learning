#!/usr/bin/env python3
""" Determine the shape of a matrix """


def matrix_shape(matrix):
    shape = [len(matrix)]
    tmp = matrix
    while isinstance(tmp[0], list):
        shape.append(len(tmp[0]))
        tmp = tmp[0]
    return shape
