# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:07:45 2020

@author: billy huang
"""
# 先判断是否是闰年,因为TRUE可以表示1，FALSE可以表示0
def year_jg(year):
    return year % 4 == 0  and year % 100 != 0 or year % 400 == 0
#建立日期表
days_of_month = [
        [31,28,31,30,31,30,31,31,30,31,30,31],
        [31,29,31,30,31,30,31,31,30,31,30,31]
        ]
# 接着判断第N天是几月几日
def day_jg(year,date):
    day = days_of_month[year_jg(year)]
    month = 1
    for i in range(12):
        if date - day[i] > 0:
            date = date - day[i]
            month += 1
        else:
            return month , date
# 开始
a = int(input("哪一年？"))
b = int(input("多少天？"))
c,d = day_jg(a,b)
print("是"+str(a)+"年"+str(c)+"月"+str(d)+"日")

"""
这里的一个小技巧在于，return返回两个值的话，可以使用两个参数把它们分出来赋值
"""

