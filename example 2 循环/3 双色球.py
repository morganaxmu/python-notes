# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:51:46 2020

@author: billy huang
"""

from random import randint, sample
def getball() :
    red_balls = [x for x in range(1,34)]
    selceted_red = sample(red_balls,6)
    selceted_red.sort()
    selceted_red.append(randint(1, 16))
    return selceted_red

def output_ball(balls):
    for index, ball in enumerate(balls):
        if index == len(balls) - 1:
            print("|",end=" ")
        print(f'{ball:0>2d}',end=" ")
    print()
    
a = int(input('抽几次？'))
for i in range(a):
    output_ball(getball())