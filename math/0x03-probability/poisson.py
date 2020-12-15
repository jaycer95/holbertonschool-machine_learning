#!/usr/bin/env python3
""" Create a class Poisson """


class Poisson:
    """  represent a poisson distribution   """

    def __init__(self, data=None, lambtha=1.):
        """ Class contructor """
        self.data = data
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))