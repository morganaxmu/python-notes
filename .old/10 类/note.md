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
