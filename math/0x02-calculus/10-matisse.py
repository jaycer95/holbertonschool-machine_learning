#!/usr/bin/env python3
"""Poly Derivate"""


def poly_derivative(poly):
    """   Calculate the derivative of a polynomial   """

    if poly == [] or not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    l = [c * p for c, p in enumerate(poly) if c]
    return l
