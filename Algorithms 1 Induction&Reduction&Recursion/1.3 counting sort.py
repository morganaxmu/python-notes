# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:14:37 2021

@author: billy huang
"""

from collections import defaultdict
def counting_sort(A, key=lambda x: x):
    B, C = [], defaultdict(list) # Output and "counts"
    for x in A:
        C[key(x)].append(x) # "Count" key(x)
    for k in range(min(C), max(C)+1): # For every key in the range
        B.extend(C[k]) # Add values in sorted order
    return B
        
M =[9,2,3,4,5,10]
print(counting_sort(M))