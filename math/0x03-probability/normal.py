#!/usr/bin/env python3
""" Create a class Normal """


class Normal:
    """  represent an Normal distribution   """

    e = 2.7182818285

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
