# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:01:50 2020

@author: billy huang
"""

import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import numpy as np
# 数据载入
df = pd.read_excel('test.xlsx')
# 曲线标绘图
x1 = np.array(df[['x1']])
x2 = np.array(df[['x2']])
x3 = np.array(df[['x3']])
pl.plot(x1,x2)
pl.plot(x1,x3)
pl.title('x1,x2&x3')
pl.xlabel('x1')
pl.ylabel('x2&x3')
pl.show()
# 连线标绘图
pl.plot(x1,x2)
pl.title('x1&x2')
pl.xlabel('x1')
pl.ylabel('x2')
pl.show()
pl.plot(x1,x2,'ro')
pl.title('x1&x2')
pl.xlabel('x1')
pl.ylabel('x2')
pl.show()