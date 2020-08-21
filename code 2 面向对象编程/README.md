> **面向对象编程**：把一组数据和处理数据的方法组成**对象**，把行为相同的对象归纳为**类**，通过**封装**隐藏对象的内部细节，通过**继承**实现类的特化和泛化，通过**多态**实现基于对象类型的动态分派。

先圈出几个关键词：**对象**（object）、**类**（class）、**封装**（encapsulation）、**继承**（inheritance）、**多态**（polymorphism）。

面向对象的三步走方法：定义类、创建对象、给对象发消息

构造一个类，如同制造一个机器。各种函数是机器内部的各种零件，用于实现各种功能。而所谓封装，就是把零件拼接起来，这样你只需要按下开关，机器就能给你想要的东西，而不需要从一个零件一个零件开始实现功能。

## 类
### 对象方法
举例如下：

```Python
class Student:

    def __init__(self, name, age):
        """初始化方法"""
        self.name = name
        self.age = age

    def study(self, course_name):
        """学习"""
        print(f'{self.name}is learning{course_name}.')

    def play(self):
        """玩耍"""
        print(f'{self.name}is playing.')
    
    def __repr__(self):
        return f'{self.name}: {self.age}'


stu1 = Student('a', 40)
print(stu1)        # a: 40
students = [stu1, Student('b', 16), Student('c', 25)]
print(students)    # [a: 40, b: 16, c: 25]
```
在Python中，可以使用`class`关键字加上类名来定义类。写在类里面的函数我们通常称之为**方法**，方法就是对象的行为，也就是对象可以接收的消息。方法的第一个参数通常都是`self`，它代表了接收这个消息的对象本身。

在类的名字后跟上圆括号就是所谓的构造器语法，当用`print`函数打印`stu1`和`stu2`两个变量时，会看到输出了对象在内存中的地址（十六进制形式）。如果在打印对象的时候不希望看到对象的地址而是看到自定义的信息，可以通过在类中放置`__repr__`魔术方法来做到，该方法返回的字符串就是用`print`函数打印对象的时候会显示的内容。

在Python中，以两个下划线`__`（读作“dunder”）开头和结尾的方法通常都是有特殊用途和意义的方法，一般称之为**魔术方法**或**魔法方法**。要给对象定义属性，需为其添加一个名为`__init__`的方法。`__init__`方法通常也被称为初始化方法，可完成对属性赋初始值的操作。

### 动态属性
```Python
class Student:

    def __init__(self, name, age):
        self.name = name
        self.age = age


stu = Student('a', 20)
# 为Student对象动态添加sex属性
stu.sex = 'male'
```
如果不希望在使用对象时动态的为对象添加属性，可以使用Python的`__slots__`魔法。对于`Student`类来说，可以在类中指定`__slots__ = ('name', 'age')`，这样`Student`类的对象只能有`name`和`age`属性，无法动态添加。

### 静态方法和类方法
静态方法和类方法就是发送给类对象的消息。对象方法是直接传递给对象，有时候会出现问题：比如现在有一个三角形对象，你可以输入abc三个数值生成三角形；但是如果abc不满足生成三角形的条件，就会扑街。因此，需要类方法，先把abc三个数值传递给三角形类，三角形类判断是否可以生成三角形之后，再传递给三角形对象。

```Python
class Triangle(object):
    """三角形类"""

    def __init__(self, a, b, c):
        """初始化方法"""
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def is_valid(a, b, c):
        """判断三条边长能否构成三角形(静态方法)"""
        return a + b > c and b + c > a and a + c > b

    # @classmethod
    # def is_valid(cls, a, b, c):
    #     """判断三条边长能否构成三角形(类方法)"""
    #     return a + b > c and b + c > a and a + c > b

    def perimeter(self):
        """计算周长"""
        return self.a + self.b + self.c

    def area(self):
        """计算面积"""
        p = self.perimeter() / 2
        return (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5
```
上面的代码使用`staticmethod`装饰器声明了`is_valid`方法是`Triangle`类的静态方法，如果要声明类方法，可以使用`classmethod`装饰器。可以直接使用`类名.方法名`的方式来调用静态方法和类方法，二者的区别在于，类方法的第一个参数是类对象本身，而静态方法则没有这个参数。

### 总结
**对象方法、类方法、静态方法都可以通过`类名.方法名`的方式来调用，区别在于方法的第一个参数到底是普通对象还是类对象，还是没有接受消息的对象**。静态方法通常也可以直接写成一个独立的函数，因为它并没有跟特定的对象绑定。

## 继承和多态

面向对象的编程语言支持在已有类的基础上创建新类，从而减少重复代码的编写。提供继承信息的类叫做父类（超类、基类），得到继承信息的类叫做子类（派生类、衍生类）。