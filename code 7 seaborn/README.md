seaborn包基于matplotlib包，其可执行的操作包括：1. 单变量分布可视化(distplot) 2. 双变量分布可视化(jointplot) 3. 数据集中成对双变量分布(pairplot) 4. 双变量-三变量散点图(relplot) 5. 双变量-三变量连线图(relplot) 6. 双变量-三变量简单拟合 7. 分类数据的特殊绘图

# 1.单变量分布
单变量分布的话，就是直接直方图hist+概率分布曲线了。如果不想要hist可以调用参数hist= False只留下分布曲线
```python
from numpy import random
x = random.rand(200)
sns.distplot(x)
```

# 2.双变量
两个变量的联合概率分布+每一个变量的分布
```python
import pandas as pd
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)
```
如果不想要hist要分布曲线，调用参数kind="kde"即可

# 3.成对多变量
直接用pairplot可以得到每一对变量之间的散点图和每个变量自己的的hist
```python
iris = sns.load_dataset("iris")
sns.pairplot(iris)
```

# 4.双变量-三变量散点图
```python
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips)
```
调用hue="变量名"可以通过颜色来引入二维图上的第三个维度，甚至可以调用style="变量名"来引入第四个维度

# 5.双变量-三变量连续图
```python
df = pd.DataFrame(dict(time=np.arange(500),value=np.random.randn(500).cumsum()))
sns.relplot(x="time", y="value", kind="line", data=df)
```
同样的，可以调用参数hue来用颜色区分

# 6.线性拟合
用replot()即可，会自带线性回归模型的直线和95%CL
```python
sns.set_style('darkgrid')
sns.regplot(x="total_bill", y="tip", data=tips)
```
当然不止一阶线性，你可以用多项式，设定order参数即可；如果担心outlier的影响，令参数robust = True即可忽略outlier
```python
anscombe = sns.load_dataset("anscombe")
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),ci=None,order = 2)
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),ci=None,robust = True)
```

# 7.特殊
## 分类分布图
即箱型图
```python
sns.catplot(x="day", y="total_bill", kind="box", data=tips)
```
依然可以调用hue来增加维度，不想要箱型图，调用参数kind："violin"-小提琴图

## 分类估计图
即条形图，bar
```python
titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
```