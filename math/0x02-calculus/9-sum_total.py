#!/usr/bin/env python 3
""" Sum squared """


def summation_i_squared(n):
    """  calculate given sum  """
    if type(n) == int and n >= 1:
        return (n * (n + 1) * (n + 2) / 6)
    return None
