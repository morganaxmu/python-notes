# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:33:05 2020

@author: billy huang
"""

# Task 2: The Guessing Game
import random
proxy = True
answer = random.randint(1,99)
while proxy:
    x = input("Enter a number:")
    if int(x) > answer:
        print("too high")
    if int(x) < answer:
        print("too low")
    else:
        print("correct!:)")
        proxy = False
