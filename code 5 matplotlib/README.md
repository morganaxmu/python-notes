## 用matplotlib数据可视化
# 1.基本参数
先导入函数
```python
import matplotlib as mpt
import matplotlib.pyplot as plt
# 通用的参数为图标标题、坐标轴标签和刻度大小
plt.plot(x,x ** 2,'-g',lable='1')
plt.title("square Numbers", fontsize=24)
plt.xlabel("Value",fontsize=14)
plt.ylabel("Square of Valuee",fontsize=14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
# 亦可设置坐标轴的取值范围
plt.axis([x1, x2, y1, y2])
plt.axis('equal')
plt.axis('tight')
plt.xlim(x1,x2)
plt.ylim(y1,y2)
# 当作图完毕后，显示图形
plt.show()
# 在Jupyter notebook中，使用下列参数（写在最前面）
%matplotlib inline
# 自动保存图标
plt.savefig('name.png', bbox_inches='tight')
# bbox_inches='tight'为裁剪掉图多余的空白
```
当然，最重要的是如果你不清空画板，直接继续作图的话，它是会在当前画板上直接继续画的。所以想要复数线在一张图上，直接继续plot就好了。但是有时候如果你在def的函数里面用plot，之后调用函数，最后可能会导致两张图挤在一起，这个时候就要在第一张图结尾的代码加一段：
```python
plt.show()
plt.clf()
```
# 2.作图函数
## （1）plot
如果直接使用.plot函数，它会自动拟合成曲线
```python
plt.plot(x, y, linewidth=5)
plt.plot(x,y, linestyle='-') # 参数可选：solid'-'，dashed'--'，dashdot'-.'，dotted':'
plt.axes
# 要改变颜色，调用color参数
# 如果需要着色，需要提供一个x和对应的y1\y2，函数会在其区间内着色
plt.plot(x,y1,c='red',alpha=0.5)
plt.plot(x,y2,c='blue',alpha=0.5)
plt.fill_between(x,y1,y2,facecolor='blue',alpha=0.1)
```
alpha是透明度，0完全透明，1完全不透明。plot和R的不同，python的plot是直接加线而不是新的一张图。

## （2）散点图
使用.scatter来绘制散点图
```
x_values = list(range(1,1001))
y_values = [x**2 for x in x_values]
plt.scatter(x_values, y_values, c=(0, 0, 0.8), edgecolor='none', s=40)
#s为点的尺寸
#2.0.0以后，默认edgecolor='none'，这一步是为了删除数据点的轮廓
#c=(0, 0, 0.8)为RGB颜色，当然也可以使用c = 'red'，或者使用颜色映射：c=y_values, cmap=plt.cm.Blues
```
## （3）直方图
```
hist = pygal.Bar()
```
相关的设置也与plot类似
```
hist.title = "Results of rolling two D6 1000 times."
hist.x_labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
hist.x_title = "Result"
hist.y_title = "Frequency of Result"
hist.add('D6+D6', frequencies)
```
保存成文件
```
hist.render_to_file('die_visual.svg')
```
当然也可以用plt包的相关函数，比如
```
import numpy as np
import matplotlib.pyplot as plt
data = np.random.randn(1000)
plt.hist(data, bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
plt.show()
```
## （4）折线图
```
import pygal
line_chart = pygal.Line(x_label_rotation=20, show_minor_x_labels=False)
#第一个参数让标签顺时针旋转20°，第二个则是不需要显示所有标签
line_chart.title = '任意标题'
line_chat.x_labels = 时间的列表
N = x坐标间隔
line_chart.x_labels_major = 时间的列表[::N]
line_chart.add('图例的名字',X的列表)
line_chart.render_to_file('文件名.svg')
```

## （5）柱状图
柱状图最简单了
```
import matplotlib.pyplot as plt

name_list= ['Monday','Tuesday','Friday','Sunday']
num_list = [1.5,0.6,7.8,6]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.show()
```
