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
array的数据类型和python自带的稍有不同，复杂许多，此处不展开。大致上依然分为bool\int\uint\float\complex。

#### 属性
array有许多属性(attribute)，比如ndim（维度数）、shape（维度的size），size（array的总size），比如：
```python
x = np.random.randint(10, size=(3,4,5))
```
其ndim为3，shape为（3，4，5），size为60。因为有三个维度，每个维度生成3、4、5个随机数

因为array的属性一致性，所以试图把float写入均为int的array的时候，会被裁剪（be truncated)成int

#### 生成
类似的array有很多生成的方法，比如：
```python
np.full((3,5),3)
```
该代码会生成三行五列的合计15个3
```python
np.arrange(0,20,2)
```
该代码会以2为步长，从0到20生成数字（实际效果为0、2、……、16、18）

类似的np.X函数还有许多，可以达到类似R里面rnom之类的效果，下面开始列举：
```python
# Create a length-10 integer array filled with zeros，创建一个长度为10的所有元素均为0的array
np.zeros(10, dtype=int)
# Create a 3x5 floating-point array filled with ones，创建一个3x5的所有元素均为1的array
np.ones((3, 5), dtype=float)
# Create a 3x5 array filled with 3.14，创建一个3x5的所有元素均为3.14的array
np.full((3, 5), 3.14)
# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)
# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)
# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))
# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))
# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
# Create a 3x2 array of random integers in the interval [0,1)
 np.random.rand(3,2)
# Create a 3x3 identity matrix
np.eye(3)
# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)
```
### 2.对array进行操作

如前文所述，array可以用常规的list的提取方法（即[start:end:step]来提取），部分方法举例如下如下：
```python
import numpy as np
big_array = np.random.rand(1000000)
print(np.sum(big_array))
print(np.max(big_array))
print(np.min(big_array))
```
调用np.X方法的运行时间会比较短，当然上述函数也可以用big_array.X()来直接调用。可以视为在np.arrary这个函数中已经设置了max\min\sum的方法，直接掉用就能获得返回值。

#### 函数操作
常见操作函数见下表，nan为考虑到缺失值Na的方法
|函数名|说明|
| :---------------: |:---------------:|
| np.sum/np.nansum | Compute sum of elements |
| np.prod/np.nanprod | Compute product of elements |
| np.mean/np.nanmean| Compute mean of elements |
| np.std/np.nanstd | Compute standard deviation |
| np.var/np.nanvar | Compute variance |
| np.min/np.nanmin | Find minimum value |
| np.max/np.nanmax | Find maximum value |
| np.argmin/np.nanargmin | Find index of minimum value |
| np.argmax/np.nanargmax | Find index of maximum value |
| np.median/np.nanmedian | Compute median of elements |
| np.percentile/np.nanpercentile | Compute rank-based statistics of elements |
| np.any | Evaluate whether any elements are true |
| np.all | Evaluate whether all elements are true |
| np.percentile(object, number) | 获取object的number分位数据 |

#### 相加 Broadcasting
array和list一样，可以进行各种加减乘除操作
```python
import numpy as np
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print(a + 5)
M = np.ones((3, 3))
print(M)
print(M + a)
```
如果dimension不同，此时加法就叫做broadcasting
```python
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a + b)
```
一般来说需要遵守下列规则，Broadcasting in NumPy follows a strict set of rules to determine the interaction between the two arrays:

** Rule 1 **: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side. 
** Rule 2 **: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
** Rule 3 **: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

如果想要快速补齐，可以使用.newaxis函数添加一个新的axis：
```python
import numpy as np
a = np.arange(3)
print(a[:, np.newaxis].shape)
M + a[:, np.newaxis]
```
broadcasting的原理举例如下：
```python
M = np.ones((2,3))
a = np.arange(3)
print(M + a)
```
它的运算过程如下：
```
M:
1,1,1
1,1,1
a:
0,1,2
-----------
先按照Rule 1，a的dimension小，所以先拉扯它的左边为1，变为（1，3）
再按照Rule 2，把1的给拉扯到2，a变成(2,3)
-----------
a:
0,1,2
0,1,2
-----------
M+a:
1,2,3
1,2,3
```
我们再来看另一个例子
```python
import numpy as np
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a)
print(b)
print(a+b)
```
```
a:
0,
1,
2,
B:
0,1,2
----------
补1，然后1变成3
---------
a:
0,0,0
1,1,1
2,2,3
b:
0,1,2
0,1,2
0,1,2
a+b:
0,1,2
1,2,3
2,3,4
```
也就是说，它把dimension不足的部分，直接复制只有1的那一个dimension，然后拓宽，拓宽到两个dimension相同再进行加法。相当于我们要如何把一位数轴上的点进行一个二维操作，比如，我们有一个一维数轴上的点（5），要对他进行二维操作（2，3），也就是x+2,y+3，我们就需要将其补成（5，5），然后再进行操作。

但是，只有1能补，如果一个是（3，2），一个是（3），后者补完变成从（1，3）变成（3，3），两边还是对不齐，就无法操作。