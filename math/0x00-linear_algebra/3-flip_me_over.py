#!/usr/bin/env python3


def matrix_transpose(matrix):
    """ transpose of a 2D matrix"""
    row = []
    m = []
    i = 0
    while i < len(matrix[0]):
        for j in matrix:
            row.append(j[i])
        m.append(row)
        row = []
        i += 1
    return m
