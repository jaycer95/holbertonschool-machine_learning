#!/usr/bin/env python3
""" Create a class Exponential """


class Exponential:
    """  represent an exponential distribution   """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Class contructor """
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """ Calculate the value of the PDF for a given period """

        if x < 0:
            return 0
        pdf = self.lambtha * (self.e ** (- self.lambtha * x))
        return pdf

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period """

        if x < 0:
            return 0
        cdf = 1 - (self.e ** (- self.lambtha * x))
        return cdf
