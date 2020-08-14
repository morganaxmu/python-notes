# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 08:06:39 2020

@author: billy huang
"""

persons = [True]*30
#每个人的编号
index = 0
#扔到海里去了几个人
count = 0
#当前传递到几
number = 0
 
while count<15:
    if persons[index]:
        number += 1
        if number == 9:
            number = 0
            count += 1
            persons[index] = False
    index += 1
    index %= 30
for person in persons:
    print('女' if person else '男', end='')    