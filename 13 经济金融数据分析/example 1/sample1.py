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
#ʹ��pandas��ȡ�Ļ�
df = pd.read_csv('./data.csv')
#�������ҳ�Ļ���data_url="net address",֮��df=pd.read_csv(data_url)
#ע��pandas�Ķ�ȡ����Ϊ����ת�������⣬Դ�ļ�����������ľ�ȫ��ת��ΪӢ��
#���ݱ任
print(df.head(n=10))
print(df.tail(n=10))
print(df.T)
#��ȡ����
print(df.ix[2:,0:3].head(n=10))
#��ȡ3~ĩβ��Ԫ�أ�����Ϊ��1\2\3�У���ȡǰ10��
#����
print(df.drop(df.columns[[2,3]],axis=1).head(n=10))
#��ӡ���˵�3��4�������е�ǰ10��Ԫ�أ����axis=0���ǵ�3��4�С�
#ͳ������
print(df.describe())
#���ӻ�
plt.show(df.plot(kind = 'box'))
plt.show(sns.boxplot(df))

