## 1.Numpy
Numpy是数据科学常用的package，在Jupyter内可以通过按下tab键获得自动补充建议或者是函数的help信息

### 1.array
array是numpy特有的类型，它的主要优势是数据类型的一致性（稍后解释），它可以用常规的list的提取方法（即[start : end : step]来提取）

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

如前文所述，array可以用常规的list的提取方法（即[start : end : step]来提取），部分方法举例如下如下：
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

## 2.Pandas
### 数据的read&write
pandas可以调用的对象类型很多，Numpy的array等都可以。对数据分析来说，较为常用的是用pandas载入数据为dataframe，就像R里面的read_csv/read_xlxs一样方便，读取函数如下所示，注意路径要与工作路径相匹配：
```python
import pandas as pd
titanic = pd.read_csv("data/titanic.csv")
#欲查看该dataframe的前8行
titanic.head(8)
```
注意pandas支持提取的时候按照条件提取，也就是说你可以用data[data.index>0]来进行提取

相应的，你也可以write
```python
import pandas as pd
titanic = pd.read_excel('titanic.xlsx', sheet_name='passengers')
titanic.to_excel('titanic.xlsx', sheet_name='passengers', index=False)
```
只需要替换read_和.to_后面的文件类型即可
### 对象类型
#### series
series很好理解，序列，就是一串数，可以把列表转换成序列：
```python
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0])
```
series可以看成一个N*2的array，其中第一列为0-N；在pandas中创建series的时候可以同时为其index命名（即改变array第一列的内容）
```python
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
```
因为这一特性，我们当然可以把key-value形式的字典也给变成series
```python
import pandas as pd
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
```
实际上，series的数据类型是index-value类型，可以通过下列代码来生成series
```python
import pandas as pd
pd.Series(5, index=[100, 200, 300])
pd.Series({2:'a', 1:'b', 3:'c'})
```
#### dataframe
数据框结构与R里面类似，你可以把它视为series的组合：
```python
import pandas as pd
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
states = pd.DataFrame({'population': population,'area': area})
```
当然，这个合并是因为两个的index都一致，所以就直接合在一起了。dataframe有两个属性，一个是index（行名），一个是columns（列名），都可以通过data.index/data..columns函数调用查看。

自然，你只要有Index和value，就能通过赋予colunms来创建dataframe
```python
import pandas as pd
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_
pd.DataFrame(population, columns=['popula
data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
```
像上述最后一个代码一样，index有abc，但是很显然0C和1A都是缺失的，那么pandas会自动补NaN上去。总而言之，只要满足index\columns\value三个条件就能创建dataframe。
#### index
index有点类似元组，它不能被改动
```python
import pandas as pd
ind = pd.Index([2, 3, 5, 7, 11])
```
当然这不意味着你不能对其进行操作，你只是不能改变其中的元素罢了
```python
import pandas as pd
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB  # intersection
indA | indB  # union
indA ^ indB  # symmetric difference
```
### 操作
#### 提取和常规操作
series只有index-value，所以按照index提取就好；如果要用数字提取也可以。

dataframe同理，但是你可以通过data['new colunms']=来创建新的columns。常见的函数如下列所示：
```python
import pandas as pd
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data.values
data.values[0]
data.T
data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']
data.loc[data.density > 100, ['pop', 'density']]
data.ix[:3, :'pop']
```
在pandas中，有三个特殊的索引函数，分别为loc\iloc\ix；loc通过行标签（index）和列标签（columns)索引，iloc通过对应坐标索引（第一个参数为行，第二个参数为列）
#### ufuncs of numpy
numpy经常和pandas一起使用，所以很多numpy的函数可以直接以pandas的series和dataframe为对象操作
```python
import pandas as pd
import numpy as np
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),columns=['A', 'B', 'C', 'D'])
np.exp(ser)
np.sin(df * np.pi / 4)
```
由于numpy的ufuncs有自动填充的功能，如果你将两个index部分重合的series/dataframe进行操作，完全没有问题，缺失的部分会用NaN填充。如果不想用NaN，可以调用pandas的函数并使用fill_value参数，如下所示：
```python
import pandas as pd
import numpy as np
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B
A.add(B, fill_value=0)
```
在pandas中，其他函数包括：
|python语法|pandas函数|
| ---- | ---- |
|+|add()|
|-|sub(), subtract()|
|*|mul(), multiply()|
|/|truediv(), div(), divide()|
|//|floordiv()|
|%|mod()|
|**|pow()|

因为dataframe相当于series的集合，所以你也可以用dataframe和series进行运算，此时可以通过调用上述函数中的axis函数来确定运算的是行还是列（默认是行，axis=0为列）

### 缺失值处理
处理缺失值通常有两种方法，mask（面具）和sentinel（哨兵），两者都会有所牺牲，Pandas采用的是后者。Pandas中的缺失值用pd.nan(import pandas as pd)来表示。

在python中，如果用None来表示缺失值，因为None的属性是object，会在执行诸如sum之类的数值运算的函数的时候发生错误；而另一种表示方法是NaN，它的属性则是float，因此可以参与计算——只不过所有结果都是NaN罢了，因此Numpy会有NaN-free的函数比如np.nansum()之类的。

但是在Pandas中不用担心，如果你把series/dataframe中的某一项赋值为None，它会自动转变成NaN，同时把所有int元素转换成float、所有boolean（布尔值）转换成object。Pandas同时还有一系列关于缺失值的函数：
|函数名|作用|
| ---- | ---- |
|isnull()|生成一个（或者系列）布尔值看表示有没有缺失值|
|notnull()|与上面一个相反|
|dropna()|返回一个去掉缺失值的data|
|fillna()|返回一个缺失值被补充的data|
```python
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()] #直接提取不是null的
data.dropna()
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df.dropna() #对dataframe使用的时候，任何行只要有null都会扔掉
df.dropna(axis='columns') # 设定axis=1或columns可以把扔掉任何有null的列
df[3] = np.nan
df.dropna(axis='columns', how='all') #如果不想全扔，设置how='all'扔掉全是null的列
df.dropna(axis='rows', thresh=3) #或者设置阈值thresh，thresh=3会保留至少有三个非null的列
# fill的方法有很多
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data.fillna(0) #全部填0
data.fillna(method='ffill') # forward-fill，填前一个值
data.fillna(method='bfill') # back-fill，填后一个值
df.fillna(method='ffill', axis=1) #也可以对dataframe使用，此时如果用ffill填充，null是第一个的话就会忽略
```
### Hierarchical Indexing分层索引
dataframe是二维的，虽然pandas还有三维的panel（对应paneldata，index-year-value三个维度），但分层索引更常用。

一般来说，可以考虑index由两个元素组成的元组tuple构成，代码如下：
```python
import pandas as pd
import numpy as np
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop[[i for i in pop.index if i[1] == 2010]]
```
但是用pandas自带的MultiIndex可以达到相同的效果
```python
index = pd.MultiIndex.from_tuples(index)
pop[:, 2010]
```
pandas也可以将多层index的series转换成dataframe，也可以反之
```python
pop_df = pop.unstack()
pop_df.stack()
```
当然，分层索引对dataframe是一样有效的
```python
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
```
创建分层索引的方法有很多。值得一提的是，当index是元组的时候，能自动创建分层索引的series
```python
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)
# 下列方法都能创建相同的multiindex
# MultiIndex(levels=[['a', 'b'], [1, 2]],
#          labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```
multiindex的.index.names属性有两个，所以切片（slice）的时候要注意；同时不止行可以有multi index，列也可以有：
```python
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```
### 合并数据集
#### 连环cancatenate和append
在Numpy中，可以使用np.concatenate()函数来将两个对象进行合并，同一维度的对象会合并在一起。在pandas中也会如此，如下所示：
```python
import pandas as pd
import numpy as np
def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')
# 不指定参数的话，默认会以行为轴合并，也就是新增行为index，列数不变；要按照列来需要axis='col'或=1
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis='col')")
# pandas的合并是会复制index/indices的，即使两个对象的index都是0，1，他也会机械地合并下去
# 如果要让index按照顺序展开而非复制之前的，需要调整参数
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # make duplicate indices!
display('x', 'y', 'pd.concat([x, y], ignore_index=True)')
# 如果要分层index，使用key参数
display('x', 'y', "pd.concat([x, y], keys=['x', 'y'])")
# 如果列的index不完全重叠，会产生NaN
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')
# 欲忽略NaN，可以调用参数join来指定保留的部分，inner只保留交集，outer反之
display('df5', 'df6',
        "pd.concat([df5, df6], join='inner')")
# 或者可以指定joinaxis，虽然亦会产生NaN
display('df5', 'df6',
        "pd.concat([df5, df6], join_axes=[df5.columns])")
# 使用append()亦可达到和concat相同的效果
display('df1', 'df2', 'df1.append(df2)')
```
#### merge&join
用pd.merge可以达到合并的效果
```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2)
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
pd.merge(df3, df4)
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
pd.merge(df1, df5)
```
df1、df2因为有共通的employee列，所以会自动按照其为基准合并；而合并df3和df4的时候，group是共通的，但是会出现一对多的情况——df3有两个engineering，那么这两个engineering都会得到supervisorGuido；当df1和df5合并的时候，因为出现了多个accounting和engineering的对应项目，则会一同合并为独立的条目，即
|employee|group|skills|
| ---- | ---- | ---- |
|Bob|Accounting|math|
|Bob|Accounting|spreadsheets|
|Jake|Engineering|coding|
|Jake|Engineering|linux|
|Lisa|Engineering|coding|
|Lisa|Engineering|linux|
|Sue|HR|spreadsheets|
|Sue|HR|organization|
在进行合并的时候，可以对参数进行调整：on参数，用于指定合并时候的基准列；如果两个df没有名字一样的列，可以通过设定left_on, right_on参数来指定基准列；同时由于合并时候，参数how默认为'inner'，合并的时候如果对不上就只会保留交集，可以通过设定为'outer'把对不上的部分变成NaN；在合并的时候，如果有columns name相同且不为基准列，pandas会默认加后缀_x和_y等来区分，如果想自己设定可以修改参数suffixes（如suffixes=["_L", "_R"]）

#### Aggregation&Grouping

