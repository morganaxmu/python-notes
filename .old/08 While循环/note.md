# 八、While循环
## 1.简述
While循环会不断运行直到指定的条件不满足为止。如果while = True，它会从头到尾循环直到给出False为止；如果while A:它会循环到给定语句条件不满足为止，比如A空了无法执行移除语句。<br>
一个不错的小技巧就是用+=X来表示迭代：A+=1等价于A=A+1<br>
例如使用while来进行input的退出：<br>
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
要移动元素，见文件夹内的confirmed_users.py<br>
要输入任意数量的信息，见文件夹内的poll.py——输入任意数量信息的技巧在于，一开始的语句A为空，而要求输入的语句为B。之后，再改变初始语句A，使其显示新的问题。要终止循环，创建一个新的输入语句C，如果其为no则终止循环。<br>
