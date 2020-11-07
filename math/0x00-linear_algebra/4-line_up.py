#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """ sum two arrays"""
    if len(arr1) != len(arr2):
        return None
    arr = [x + y for x, y in zip(arr1, arr2)]
    return arr
