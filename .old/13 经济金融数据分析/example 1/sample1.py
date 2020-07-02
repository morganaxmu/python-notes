#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sample1.py
#  
#  Copyright 2020 billy huang <billy huang@DESKTOP-77CQ0AV>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#使用pandas读取的话
df = pd.read_csv('./data.csv')
#如果是网页的话，data_url="net address",之后df=pd.read_csv(data_url)
#注意pandas的读取，因为编码转换的问题，源文件内如果有中文就全部转换为英文
#数据变换
print(df.head(n=10))
print(df.tail(n=10))
print(df.T)
#提取数据
print(df.ix[2:,0:3].head(n=10))
#提取3~末尾的元素，对象为第1\2\3列，提取前10个
#舍弃
print(df.drop(df.columns[[2,3]],axis=1).head(n=10))
#打印除了第3，4列外所有的前10个元素，如果axis=0则是第3，4行。
#统计描述
print(df.describe())
#可视化
plt.show(df.plot(kind = 'box'))
plt.show(sns.boxplot(df))

