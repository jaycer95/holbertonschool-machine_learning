#!/usr/bin/env python3
""" Integrate Poly  """


def is_whole(n):
    """ Check if number is whole """
    return n % 1 == 0


def poly_integral(poly, C=0):
    """ Calculate the integral of a polynomial """
    lp = [C]
    if not isinstance(C, (int, float)):
        return None
    if not isinstance(poly, list) or poly == []:
        return None
    for i in range(len(poly)):
        if is_whole(poly[i] / (i + 1)):
            lp.append(int(poly[i] / (i + 1)))
        else:
            lp.append(poly[i] / (i + 1))
    return lp
