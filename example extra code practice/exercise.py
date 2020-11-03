# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:17:02 2020

@author: billy huang
"""
import numpy as np
from numpy import random
import timeit
import math
import doctest

random.seed(20201103)

def timer(func):
    starttime = timeit.default_timer()
    print(func)
    print("The time difference is :", timeit.default_timer() - starttime)

# task 1, function, finding the max
def findmax(l):
    """
    >>> findmax([8,2,4,4,5,6,7,8])
    8
    """
    max = 0
    for i in range(0,len(l)):
        if l[i]>max:
            max = l[i]
    return max
doctest.testmod()

x = random.rand(10)
timer(findmax(x)) # testing the time

# task 2
import pandas as pd
from pandas.core.frame import DataFrame
timer(x.sort())

timer(np.array(x).sort()) # numpy

y = DataFrame(x)
y.columns = ['col1']
timer(DataFrame(y).sort_values('col1')) # pandas


# task 3
def find_up_q(l,q):
    """
    >>> find_up_q([1,2,3,4,5,6,7,8,9,10],0.75)
    7.75
    """    
    i = int(1+(len(l)-1)*q)
    j = 1+(len(l)-1)*q
    sorted_l = sorted(l)
    return sorted_l[i-1] + (sorted_l[i]-sorted_l[i-1])*(j-i)
doctest.testmod()

timer(find_up_q(l=x, q=.75))

timer(np.quantile(np.array(x),0.75)) # numpy

timer(y.quantile(.75)) # pandas

# task 4

