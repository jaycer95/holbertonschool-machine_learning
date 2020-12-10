#!/usr/bin/env python3
""" Integrate Poly  """


def is_whole(n):
    """ Check if number is whole """
    return n % 1 == 0


def poly_integral(poly, C=0):
    """ Calculate the integral of a polynomial """
    lp = [C]
    if not isinstance(C, (int, float))
    or not isinstance(poly, list) or poly == []:
        return None
    for i in range(0, len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        try:
            if is_whole(poly[i] / (i + 1)):
                lp.append(int(poly[i] / (i + 1)))
            else:
                lp.append(poly[i] / (i + 1))
        except ZeroDivisionError:
            lp.append(0)
    return lp
