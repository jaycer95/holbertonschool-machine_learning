#!/usr/bin/env python3
"""Poly Derivate"""


def poly_derivative(poly):
    """   Calculate the derivative of a polynomial   """
    list = []
    if poly is None:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        list.append(i * poly[i])
    return list