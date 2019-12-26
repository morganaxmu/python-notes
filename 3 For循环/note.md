# 三、For循环
循环：对于一个列表As循环如下：
```
for A in As:
   function(A)
```
对于字典：
```
for key, value in A.items():
   function(key)
   function(value)
```
要让Python理解到这是一个循环，for之后的下一行要缩进。利用range()+循环可以生成列表（Python中两个乘号表示乘方）
```
for value in range(1,11):
   square=vlue**2
   squares.append(square)
```
注意这里range(1,11)是1-10，11并不执行。range有第三个参数，默认为1,range(1,11,2)会生成1，3，5，7，9按照2的步长生成结果。
或者使用列表解析进行合成：
```
squares=[value**2 for value in range(1,11)]
```
对数字列表，还有常用min(),max(),sum()方法
