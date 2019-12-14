# python-notes
Some notes when I was learning Python
# Index
杂项	
 1.配置环境变量	
 2.使用pip安装包	
一、字符串和数字	
二、列表	
三、For循环	
四、元组	
五、If语句	
六、字典	
七、用户输入	
八、While循环	
 1.简述	
 2.标志	
 3.循环的退出与继续	
 4.利用循环处理列表和字典	
九、函数	
 1.定义函数	
 2.传递实参	
 （1）位置实参	
 （2）关键字实参	
 （3）默认值	
 3.返回值	
十、类	
 1.创建类	
 2.继承	
 3.以实例作为属性	
 4.导入类	
十一、文件和异常	

# 杂项
## 1.配置环境变量
不配置环境变量就无法使用python。
首先我们来到python的安装目录（默认是c盘）然后打开python27下面的scripts目录，然后复制这个路径 
右键我的电脑点击属性，然后点击高级系统设置，再点击环境变量在系统变量里面有个path打开他，然后把刚刚复制的路径粘贴进去（注意 path里面的东西不要删），然后一直保存就可以了 。
## 2.使用pip安装包
一般python3默认带了pip3，就在python的scripts目录下。
要使用pip，只需要在配置了环境变量之后，打开cmd，然后输入pip即可。
python –m pip install –upgrade pip
python –m pip install request
## 3.杂项的数据清洗
比如爬虫获取到的是字典，但是里面镶嵌的是列表
输出成txt，单独复制出来列表，用word打开，把“[”换成空格，“],”换成换行符（WORD的替换才能做到，txt的文本编辑器不行）然后再变成CSV文件
<br>
#一、字符串和数字
Python中，””，’’括起来的就是字符串，如果字符串中包含’，应当使用””作为括号。字符串区分大小写，转换大写：用X.upper()，转换小写用X.lower()，只要首字母大写用X.title()
Python使用+来合并字符串。例如：message=”Hello”+” ”+”world”，\n换行\t制表符也就是首行缩进。Python对空白敏感，去除不必要空白的方法是.rstrip()，但该方法不改变原定义，要去除原定义中的空白，使用A=A.rstrip()
Python对数据类型敏感，数字用str()转化为字符串。字符串用int()转换为数值
```
编写注释#，使用方法与R一致。
+-*/与R一致，**是乘方，%是求模：两个数相除并返回余数
注意：python是UTF-8编码！不能包含任何的中文！连注释也一样！
```

# 二、列表
Python中的方法调用，是直接在变量后面+.function，比如，提取A的第一个元素并且将首字母大写：A[0].title()，()内是方法的参数，与R相似。
Python中，第一个元素编号为0，即所有元素的序号都-1.与R相似，Python中，访问某一元素（提取某一元素）使用中括号[]，比如提取A的第二个元素，就是A[1]。当然-1也可以。当然也可以使用“切片”进行提取，即[0:3]提取0、1、2三个元素，如果不指定第一个数字则默认从头开始提取；如果不指定最后一个元素，则默认提取到末尾。当然也可以使用-3之类的负数。一个复制列表的小技巧就是B=A[:]既不指定第一个索引也不指定最后一个索引。
列表：列表通常表示为[“A”，“B”]用中括号表示列表，每个元素用””或者’’括起来。在列表末尾添加元素，使用X.append(“元素”)；插入元素使用X.insert(位置，”元素”)；删除元素使用del X[位置]以永久删除，使用X.pop(位置)删除元素，不写位置默认为最后一个元素，该元素会从原列表消失，但可以使用X.pop(位置)调取该元素。如果要根据值删除元素，使用X.remove(“值”)
使用sort()使元素按字母顺序A-Z排序（全为小写或大写），参数reverse=True以反序排序。sorted()可以进行临时排序。反转排列顺序，直接使用X.reverse()

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

# 四、元组
Python中元组不可修改，与列表不同元组使用()而非[]

# 五、If语句
Python中也有if语句。用==来检查条件相等，即条件测试：A==B 如果相等返回True，反之返回False，检查时区分大小写。用!=检查条件不相等，!表示不。合并测试条件用and，如age >= 21 and life >=22
检查是否在列表中，只需要输入 not in 
e.g.: if A not in As:
布尔值即为True和False，可用于记录条件。
If语句由条件测试和执行语句构成：
```
if 条件测试：
  执行语句
elif 条件测试：
  执行语句：
else：
  执行语句
 ```
else是可以被忽略的，此时仅执行通过测试的语句。在if-elif-else结构中，有一个通过检验则其它都不会被检查，所以如果不是互斥关系，应当使用独立的if 语句。

# 六、字典
Python中字典与Json相同，都是键值对形式：A={‘key’:’Value’,…….}。提取字典中的value使用X[‘key’]进行提取。添加键值对，只需要X[key]=’value’即可。修改只需要X[key]=’value’即可。删除使用del X[‘key’]
为使键值对看上去好看，最后使用如下格式：
 ```
X={
‘key1’=’value1’,
‘key2’=’value2’
}
 ```
调用所有的key只需要使用X.keys()即可。调用所有的value只需要使用X.values()即可
可以把一系列字典嵌套在列表中，只需要分别创建字典，之后X=[a,b,c]即可。同理，把列表嵌套在字典中，只需要把value变成列表即可。

# 七、用户输入
如果要让用户输入，使用input(‘提示’)，之后用户输入的值就会被赋予它，所以最好A=input(‘提示’)，之后调用A就能得到用户输入的值。如果要提示太长要换行，可以使用:
 ```
A = input(‘提示前半’)
A += “\n提示后半 ”   #此处留个空格避免用户输入与提示挤在一起
注意：在while循环等有首行缩进的语句中，无法使用 +=连在一起
 ```

# 八、While循环
## 1.简述
While循环会不断运行直到指定的条件不满足为止。如果while = True，它会从头到尾循环直到给出False为止；如果while A:它会循环到给定语句条件不满足为止，比如A空了无法执行移除语句。
一个不错的小技巧就是用+=X来表示迭代：A+=1等价于A=A+1
例如使用while来进行input的退出：
 ```
promt=’\nXXXXX’
promt  += “\nEnter ’quit’ to end the program.”
message=” ” # message一开始一定要有东西不能没有，没有任何东西会导致首次运行出错。
while message != ‘quit’:
  message= input(promt)
  if message != ‘quit’ #增加一个检验以避免quit时打印出quit
    print(message)
 ```
## 2.标志
标志用于判断程序是否应该继续进行，如：
 ```
promt=’\nXXXXX’
promt  += “\nEnter ’quit’ to end the program.”
active = True #active就是一个标志
while active:
   message=input(promt)
   if message==’quit’
     active=False
   else:
     print(message)
 ```
## 3.循环的退出与继续
可以使用break语句退出循环
 ```
promt=’\nXXXXX’
promt  += “\nEnter ’quit’ to end the program.”
while True:
  city=input(prompt)
  if city == ‘quit’:
    break
  else:
    print(‘XXX’)
 ```
使用continue利用条件测试继续循环
 ```
    cn=0
    while cn < 10
       cn+=1
       if cn %2 ==0:
         continue #continue直接返回循环的开始，进而不执行余下语句
       print(cn)
 ```
## 4.利用循环处理列表和字典
要移动元素，见文件夹内的confirmed_users.py
要输入任意数量的信息，见文件夹内的poll.py——输入任意数量信息的技巧在于，一开始的语句A为空，而要求输入的语句为B。之后，再改变初始语句A，使其显示新的问题。要终止循环，创建一个新的输入语句C，如果其为no则终止循环。

# 九、函数
## 1.定义函数
 ```
def greet_user():
    """show a single hello"""
    print('Hello')
    
greet_user()
 ```
如上述，使用def来定义一个函数，该函数使用”””docstring文档字符串”””作为注释。调用函数时只需要使用该函数即可。
如果要设置参数，只需要：
 ```
def greet_user(A):
    """show a single hello"""
    print('Hello'+A)
    
greet_user(B)
 ```
这里面A就是参数。A是一个形式参数，简称形参；而下面的B是一个实参。实参是调用函数时传递给函数的信息。实参要放在括号内。
## 2.传递实参
### （1）位置实参
位置实参：要求实参的顺序和形参的顺序相同
 ```
def greet_user(X,Y):
    """show a single hello"""
    print('Hello'+X+Y)
    
greet_user(A,B)
 ```
A执行X，B执行Y，这与我们传统的函数参数设置相同。

### （2）关键字实参
关键字实参：每个实参都由变量名和值组成
 ```
def greet_user(X,Y):
    """show a single hello"""
    print('Hello'+X+Y)
    
greet_user(X=’A’,Y=’B’)
 ```
### （3）默认值
 ```
def greet_user(X,Y=’B’):
    """show a single hello"""
    print('Hello'+X+Y)
    
greet_user(X=’A’)或者greet_user(A)
 ```
这里Y已经有默认值B，所以只输入一个实参会默认给X。
如果默认值为空，即Y=’’，就可以让用户可以选择不提供该参数。
## 3.返回值
使用return A来返回A的值。调用返回值的函数时，需要调用一个变量用于储存返回的值。

# 十、类
## 1.创建类
类中的函数称为“方法”
使用如下命令创建类：
 ```
Class Dog(): #按照一般的规范，类的首字母要大写；而实例一般首字母小写
	“””XXX”””
	
	def  __init__(self,name,age): #自动运行的默认方法，前后各加两个下划线
	self.name = name
	self.age = age
	
	def sit(self):
		“””YYY”””
		print(self.name.title()+” is now sitting.”)
	
	def roll_over(self):
		“””ZZZ”””
		print(self.name.title()+” rolled over!”)
 ```
在上述例子中，形参self必不可少并且一定要位于其它形参前面。因为调用该默认方法时将自动传入实参self，它是一个指向实例本身的引用，让实例能够访问类中的属性和方法。
之后的过程中，都不需要为形参self赋予实参。以self为前缀的变量都可以供类中的所有方法使用，我们还可以通过类的任何实例来访问这些变量。这些可以通过实例访问的变量称为属性（比如这里的self.name,self.age，他们的初始值是形参name,age对应的实参）。
这里的逻辑是这样的：self有许多属性，self.name = name的作用是在实例中把形参对应的实参赋予到self的这个属性中去。当然也可以创建没有形参、但后续方法会用到它的属性，比如self.odo=0，之后的方法调用该属性一样使用self.odo调用即可。
要修改属性，直接在后续方法中给定一个形参，然后把该形参赋值给目标属性即可。要递增的话，使用缩写型+=即可。
在实例中调用类，使用：
 ```
	my_dog = Dog(‘A’,6)
   ```
之后可以使用my_dog.name调用’A’，用my_dog.age调用6——使用X.属性 来调用属性的值。要调用方法，使用my_dog.sit()。
## 2.继承
一个类继承另一个类时，它将自动获得另一个类的所有属性和方法；原有的类称为父类，而新类称为子类。子类继承了其父类的所有属性和方法，同时还可以定义自己的属性和方法。
 ```
Class Dog(): #按照一般的规范，类的首字母要大写；而实例一般首字母小写
	“””XXX”””
	
	def  __init__(self,name,age): #自动运行的默认方法，前后各加两个下划线
	self.name = name
	self.age = age
	
	def sit(self):
		“””YYY”””
		print(self.name.title()+” is now sitting.”)
	
	def roll_over(self):
		“””ZZZ”””
		print(self.name.title()+” rolled over!”)
#在创建子类时，父类必须包含在当前文件中且位于子类之前
class Tog(Dog): #创建子类，方法为 class 子类名(父类名)
	“””ZZZZZZ”””
	Def __init__(self,name,type):
		“””初始化父类的属性”””
		Super().__init__(name,type)
 ```
	super()是一个特殊函数，调用父类的方法__init__，使子类包含父类所有的属性。父类也成为超类（superclass），因此得名super()
可以在子类中定义新方法。如果新方法与父类中原有方法同名，则替换从父类中继承的方法（但你调用父类的时候不变）。
## 3.以实例作为属性
只需要先创建一个类A，然后在类B的__init__中设置 一个属性（该属性可以与B的形参无关），其赋值为类A，这样只要使用类B创建实例（也就是eg=B(XX)），该实例就会自动赋予类A中属性数值，而不需要重复用类A创建实例（也就是eg2=A(XX)）。
## 4.导入类
语句：from 文件名 import 类名
例如：from car import Car
 ```
如果要导入多个类，用逗号分隔开个各类:A, B, C；如果要导入所有模块，使用*即可
如果要导入整个模块，，语句： import 文件名
 ```

# 十一、文件和异常
## 1.从文件中读取数据
读取语法：
 ```
with open(‘相对路径名’) as name:
 ```
相对路径名，为在当前执行文件的目录下该文件的名字，如test.txt
如果要读取文件的内容，用 name.read()方法加载，最好将之赋值给一个变量。但read()方法读取到末尾会出现一个空行，如果不需要可以使用.rstrip()方法
使用绝对路径名时，最好先用变量（比如path）储存该绝对路径，如E:\My Documents\R
如果要按行读取，可以用for 循环： 
 ```
for line in name ():  #或者先创建一个变量，将name.readlines()赋值给它
 ```
执行语句
如果要把所有行整合到一行，直接累加即可。
## 2.写入文件
语句：
 ```
filename = ‘文件名称.后缀’

with open(filename,’w’) as name:
	name.write(‘XXX’)
 ```
open函数中第二个实参w表示写入模式，r为读取模式，a为附加模式，r+为读取和写入模式。如果缺省，默认用只读打开；如果写入的文件不存在，函数open()会自动创建它。
使用w模式打开文件时，如果源文件存在，会清空该文件。
.write方法不会自动换行，要自己输入换行符、空格、制表符
如果要添加内容而不是覆盖原有内容，应当使用a模式。
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
如果执行语句1没有运行错误，那么就会无视except之后的语句，接着执行语句3；如果运行出错，且错误类型一致，就会执行执行语句2.
使用异常能够有效的避免程序因为错误而崩溃——异常让它能够继续运行，而不是工作到一半就崩溃。
例子见division.py
如果不想在出错时执行任何语句，使用pass即可。
## 4.储存数据
最简单的储存方法是储存为JSON格式。
使用前首先要载入JSON，语句：import json
一个储存的方法是写入文件中，使用json.dump语句：json.dump(对象,写入的文件名)，如：
 ```
Import json

username=input(“what is your name?”)

filename=’username.json’
with open(filename,’w’) as f_obj:
	json.dump(username,f_obj)
 ```
要从json中加载，使用json.load(文件名)即可。
## 5.重构代码
重构代码，是将代码划分为一系列完成具体工作的函数。
比如，要实现功能：“如果之前储存了该用户名，则向该用户发送问候消息；如果没有则提示用户创建一个新的”可以分为三块：
第一块，检测有没有储存该用户名：如果有，返回username；如果没有，返回none；
第二块，创建新的用户名，储存在文件中后，返回username。
第三块，问候用户。调用第一块的函数，如果返回none调用第二块；如果返回username则进行问候。
