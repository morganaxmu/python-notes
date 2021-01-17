# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:11:29 2021

@author: billy huang
"""

def naive_max_perm(M, A=None):
    if A is None: # The elt. set not supplied?
        A = set(range(len(M))) # A = {0, 1, ... , n-1}
    if len(A) == 1: return A # Base case -- single-elt. A
    B = set(M[i] for i in A) # The "pointed to" elements
    C = A - B # "Not pointed to" elements
    if C: # Any useless elements?
        A.remove(C.pop()) # Remove one of them
        return naive_max_perm(M, A) # Solve remaining problem
    return A # All useful -- return all

M = [2, 2, 0, 5, 3, 5, 7, 4]
print(naive_max_perm(M))

def max_perm(M):
    n = len(M) # How many elements?
    A = set(range(n)) # A = {0, 1, ... , n-1}
    count = [0]*n # C[i] == 0 for i in A
    for i in M: # All that are "pointed to"
        count[i] += 1 # Increment "point count"
    Q = [i for i in A if count[i] == 0] # Useless elements
    while Q: # While useless elts. left...
        i = Q.pop() # Get one
        A.remove(i) # Remove it
        j = M[i] # Who's it pointing to?
        count[j] -= 1 # Not anymore...
        if count[j] == 0: # Is j useless now?
            Q.append(j) # Then deal w/it next
    return A # Return useful elts.

print(max_perm(M))