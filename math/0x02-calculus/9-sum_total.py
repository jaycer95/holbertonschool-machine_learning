#!/usr/bin/env python 3
""" Sum squared """


def summation_i_squared(n):
    """  calculate given sum  """
    sum = 0
    if type(n) == int and n >= 1:
        for i in range(1, n+1):
            sum += i ** 2
        return sum
    return None
