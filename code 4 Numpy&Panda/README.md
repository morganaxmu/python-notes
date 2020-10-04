## 1.Numpy
Numpy是数据科学常用的package，在Jupyter内可以通过按下tab键获得自动补充建议或者是函数的help信息

### 1.array
array是numpy特有的类型，它的主要优势是数据类型的一致性（稍后解释），它可以用常规的list的提取方法（即[start:end:step]来提取）

np.array是一个比较常用的生成集合的函数
```python
import numpy as np
np.array([3.14,1,2,3])
list(3.14,1,2,3)
```
和list不同，array生成的集合内数据类型一致，3.14是float所以后面的均为float；而直接使用list生成的话，3.14是float，其他均为int。

如果想设定其中的数据类型，可以调用参数dtype：
```python
import numpy as np
np.array([3.14,1,2,3],dtype='float32')
```
类似的array有很多生成的方法，比如：
```python
np.full((3,5),3)
```
该代码会生成三行五列的合计15个3
```python
np.arrange(0,20,2)
```
该代码会以2为步长，从0到20生成数字（实际效果为0、2、……、16、18）

类似的np.X函数还有许多，可以达到类似R里面rnom之类的效果

array有许多属性(attribute)，比如ndim（维度数）、shape（维度的size），size（array的总size），比如：
```python
x = np.random.randint(10, size=(3,4,5))
```
其ndim为3，shape为（3，4，5），size为60。因为有三个维度，每个维度生成3、4、5个随机数

因为array的属性一致性，所以试图把float写入均为int的array的时候，会被裁剪（be truncated)成int

