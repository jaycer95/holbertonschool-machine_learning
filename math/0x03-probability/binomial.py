#!/usr/bin/env python3
""" Create a class Binomial """


class Binomial:
    """  represent an Binomial distribution   """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """ Class contructor """
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")

            """ Normal approximation """
            mean = float(sum(data) / len(data))
            a = 0
            for i in data:
                a += (i - mean) ** 2
            var = (a / len(data))
            self.n = round(mean ** 2 / (mean - var))
            self.p = mean / self.n
