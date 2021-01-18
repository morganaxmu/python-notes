# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:22:11 2021

@author: billy huang
"""

def naive_topsort(G, S=None):
    if S is None: S = set(G) # Default: All nodes
    if len(S) == 1: return list(S) # Base case, single node
    v = S.pop() # Reduction: Remove a node
    seq = naive_topsort(G, S) # Recursion (assumption), n-1
    min_i = 0
    for i, u in enumerate(seq):
        if v in G[u]: min_i = i+1 # After all dependencies
    seq.insert(min_i, v)
    return seq

def topsort(G):
    count = dict((u, 0) for u in G) # The in-degree for each node
    for u in G:
        for v in G[u]:
            count[v] += 1 # Count every in-edge
    Q = [u for u in G if count[u] == 0] # Valid initial nodes
    S = [] # The result
    while Q: # While we have start nodes...
        u = Q.pop() # Pick one
        S.append(u) # Use it as first of the rest
        for v in G[u]:
            count[v] -= 1 # "Uncount" its out-edges
            if count[v] == 0: # New valid start nodes?
                Q.append(v) # Deal with them next
    return S