## 十二、数据可视化
# 1.基本参数
先导入函数
```
import matplotlib.pyplot as plt
```
通用的参数为图标标题、坐标轴标签和刻度大小
```
plt.title("square Numbers", fontsize=24)
plt.xlabel("Value",fontsize=14)
plt.ylabel("Square of Valuee",fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
```
亦可设置坐标轴的取值范围
```
plt.axis([x1, x2, y1, y2])
```
当作图完毕后，显示图形
```
plt.show()
```
如果要自动保存图标
```
plt.savefig('name.png', bbox_inches='tight')
# bbox_inches='tight'为裁剪掉图多余的空白
```
# 2.作图函数
## （1）plot
如果直接使用.plot函数，它会自动拟合成曲线
```
plt.plot(x, y, linewidth=5)
```
如果不提供数据点，默认第一个数据点的x坐标为0
如果需要着色，需要提供一个x和对应的y1\y2，函数会在其区间内着色
```
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
