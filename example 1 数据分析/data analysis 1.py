# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 07:30:08 2020

@author: billy huang
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 数据载入
df = pd.read_excel('test.xlsx')
print(df.head(n=15))
X = np.array(df[['x1']])
print(X.shape)
# 直方图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(X, bins=4)
plt.title('distribution of x1')
plt.xlabel('owned games')
plt.ylabel('number of people')
plt.show()
# 参数说明：bins：直方图中箱子 (bin) 的总个数。个数越多，条形带越紧密。
# color：箱子的颜色。
# normed：对数据进行正则化。决定直方图y轴的取值是某个箱子中的元素的个数 
# (normed=False), 还是某个箱子中的元素的个数占总体的百分比 (normed=True)。
# 散点图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['x1'],df['x2'])
plt.title('x1&x2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# x,y：数组。
# s：散点图中点的大小，可选。
# c：散点图中点的颜色，可选。 
# marker：散点图的形状，可选。
# alpha：表示透明度，在 0~1 取值，可选。
# linewidths：表示线条粗细，可选。
# 气泡图，用散点图中S参数的变化来形成大小不一的气泡
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['x1'],df['x2'],s = df['x3'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# 箱图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(df['x1'])
plt.title('box figure of x1')
plt.show()
# 多个数据的箱图
vars = ['x1','x2','x3']
data = df[vars]
plt.show(data.plot(kind='box'))
# 条形图
var = df.groupby('gender').x1.sum()
fig =plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('gender')
ax1.set_ylabel('sum of x1')
ax1.set_title('gender wise sum of x1')
var.plot(kind='bar')
# 曲线图
