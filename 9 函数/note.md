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
