#!/usr/bin/env python3


def matrix_transpose(matrix):
    """ transpose of a 2D matrix"""
    m = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return m
