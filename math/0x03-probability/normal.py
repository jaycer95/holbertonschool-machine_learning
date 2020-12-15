#!/usr/bin/env python3
""" Create a class Normal """


class Normal:
    """  represent an Normal distribution   """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Class contructor """
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            a = 0
            for i in data:
                a += (i - self.mean) ** 2
            self.stddev = (a / len(data)) ** (1/2)

    def z_score(self, x):
        """ Calculate the z-score of a given x-value """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculate the x-value of a given z-score """

        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculate the value of the PDF for a given x-value """

        p1 = 1 / (self.stddev * (2 * self.pi) ** 1/2)
        p2 = self.e ** (-(x - self.mean) ** 2 / (2 * self.stddev ** 2))
        return p1 * p2
