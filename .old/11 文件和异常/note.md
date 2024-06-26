# 十一、文件和异常
## 1.从文件中读取数据
读取语法：
 ```
with open('相对路径名') as name:
 ```
 e.g.
 ```
 filename = 'xxx.csv'
 with open(filename) as f:
 	reader = csv.reader(f)
	header_row = next(reader)
	print(header_row)
```
上面例子中的代码，读取了文件的第一行并储存在header_row中。<br>
相对路径名，为在当前执行文件的目录下该文件的名字，如test.txt<br>
如果要读取文件的内容，用 name.read()方法加载，最好将之赋值给一个变量。但read()方法读取到末尾会出现一个空行，如果不需要可以使用.rstrip()方法<br>
使用绝对路径名时，最好先用变量（比如path）储存该绝对路径，如E:\My Documents\R <br>
如果要按行读取，可以用for 循环： <br>
 ```
for line in name ():  #或者先创建一个变量，将name.readlines()赋值给它
 ```
执行语句
如果要把所有行整合到一行，直接累加即可。
## 2.写入文件
语句：
 ```
filename = '文件名称.后缀'

with open(filename,'w') as name:
	name.write('XXX')
 ```
open函数中第二个实参w表示写入模式，r为读取模式，a为附加模式，r+为读取和写入模式。如果缺省，默认用只读打开；如果写入的文件不存在，函数open()会自动创建它。<br>
使用w模式打开文件时，如果源文件存在，会清空该文件。<br>
.write方法不会自动换行，要自己输入换行符、空格、制表符。<br>
如果要添加内容而不是覆盖原有内容，应当使用a模式。<br>
此外，因为write只能保存string，所以一般要用str()把数据转换成string形式。<br>
有的时候会出现如下错误：<br>
```
'gbk' codec can't encode character
```
此时只需要在open后面加一个编码就行，如：
```
with open(filename,'w', encoding='utf-8') as name:
   name.write(str(content))
```
有的时候，会出现Python爬虫解析json遇到一类JSONDecodeError<br>
此时，直接使用read()即可<br>
```
with open(filename) as f:
   data = f.read()
```
当然，如果要把list或者dict写入CSV，最简单是调用csv库，这样就可以规避write所需要的string限制<br>
```
import csv
with open(filename1, 'w', newline='') as cf1:
	csv1 = csv.writer(cf1)
	csv1.writerow(times)
```
## 3.异常
异常使用try-except代码块，其语句结构为：
 ```
try:
	执行语句1
except 错误类型（如ZeroDivisionError，FileNotFoundError）:
	执行语句2
else:
	执行语句3
 ```
如果执行语句1没有运行错误，那么就会无视except之后的语句，接着执行语句3；如果运行出错，且错误类型一致，就会执行执行语句2.<br>
使用异常能够有效的避免程序因为错误而崩溃——异常让它能够继续运行，而不是工作到一半就崩溃。<br>
例子见division.py<br>
如果不想在出错时执行任何语句，使用pass即可。<br>
## 4.储存数据
最简单的储存方法是储存为JSON格式。<br>
使用前首先要载入JSON，语句：import json <br>
一个储存的方法是写入文件中，使用json.dump语句：json.dump(对象,写入的文件名)，如：<br>
 ```
Import json

username=input("what is your name?")

filename='username.json'
with open(filename,'w') as f_obj:
	json.dump(username,f_obj)
 ```
要从json中加载，使用json.load(文件名)即可。<br>
保存json文件的时候，其无法直接保存爬虫抓下来的字典，需要先转换成string <br>
```
strhtml=requests.get(url,headers=headers,verify=False) 
content = json.loads(strhtml.text)
filename = 'Gems.json'
with open(filename,'w',encoding='utf-8') as name:
   name.write(str(content))
```
这导致在读取其的时候，要转换回字典。否则python会继续认为json中的元素为string
```
filename = 'Gems.json'
with open(filename) as f:
   data = f.read()
   data = eval(data)
```
或者，需要读取数据进一步处理时，则需要使用int()函数将其转换为整数数值，否则无法处理。但是int()只能处理整数，如果有小数点后的数值，需要先用float()转换，再用int()。<br>
## 5.重构代码
重构代码，是将代码划分为一系列完成具体工作的函数。<br>
比如，要实现功能：“如果之前储存了该用户名，则向该用户发送问候消息；如果没有则提示用户创建一个新的”可以分为三块：<br>
第一块，检测有没有储存该用户名：如果有，返回username；如果没有，返回none；<br>
第二块，创建新的用户名，储存在文件中后，返回username。<br>
第三块，问候用户。调用第一块的函数，如果返回none调用第二块；如果返回username则进行问候。<br>
