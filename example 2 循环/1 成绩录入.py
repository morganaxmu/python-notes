# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:48:48 2020

@author: billy huang
"""
import pandas as pd
df = pd.read_excel('test.xlsx')
"""
我们先根据之前的成绩继承名字创建一个新的表，用于录入成绩
先提取name出来，并重新设置科目，以此用循环生成新的scores

"""
names = df['name']
subjects = ['midterm','final','assignment']
scores = [[0]*len(subjects) for i in range(len(names))]
# 录入数据
for i, name in enumerate(names):
    print(f'请输入{name}的成绩 ===>')
    for j, subject in enumerate(subjects):
        a = input(f'{subject}: ')
        if float(a) <=100 :
                scores[i][j] = float(a)
        else:
                print('数值非法')
                break
print()
print('-' * 5, '学生平均成绩', '-' * 5)
# 计算每个人的平均成绩
for index, name in enumerate(names):
    avg_score = sum(scores[index]) / len(subjects)
    print(f'{name}的平均成绩为: {avg_score:.1f}分')
print()
print('-' * 5, '课程平均成绩', '-' * 5)
# 计算每门课的平均成绩
for index, subject in enumerate(subjects):
    # 用生成式从scores中取出指定的列创建新列表
    curr_course_scores = [score[index] for score in scores]
    avg_score = sum(curr_course_scores) / len(names)
    print(f'{subject}的平均成绩为：{avg_score:.1f}分')