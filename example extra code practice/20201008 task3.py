# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:33:25 2020

@author: billy huang
"""

# Task 3: Prime Factorisation
def primefac(number):
    factors = []
    while number > 1:
        for i in range(number-1):
            k = i + 2
            if number % k == 0:
                factors.append(k)
                number = int(number/k)
                break
    return factors

proxy = True
while proxy:
    x = input("Enter a number between 2 to 100,000,000:")
    x = eval(x)
    if x < 2 or x > 100000000 :
        print("Enter a number between 2 to 100,000,000 !!!!!!!!!")
    else:
        for i in primefac(x):
            print(str(i) + '*' ,end=' ')
        proxy = False

    

