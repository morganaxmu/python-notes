# k-mean 思路说明
## 看一眼数据
首先是导入数据，这没什么好说的，pandas的读取就是了

之后先观察每个变量的单独分布情况，用subplot分三个区域，然后用distplot即可画三个hist出来。在代码的title处使用了格式化输出，注意format(x)替代的是{}处的文本

当然，我们也可以用pairplot直接整出每个变量的hist+两两散点图。需要注意的是此处需要用两个中括号提取出dataframe的数据，一个中括号会出现keyerror的错误

接下来做简单回归。考虑到有多个变量，所以就采用循环来画。plt.subplot以及用n来指代图的位置，老一套了，不多加赘述。regplot在指定了data之后可以直接用string作为x,y扔进去，不加赘述。需要注意的是最后的ylabel，xlabel自动出来没啥好说的，但是y这边不是很想让后面的(k$)这种东西跟着一起出来，所以用.split。split函数默认按照空格把string拆成一个list。套一个简单的循环来指定三个三个ylabel

最后分男女看看两两一组的趋势，data在指定的时候，利用循环+flitter条件dataset[dataset['Gender'] == gender来筛选。

## k-mean部分
第一步当然是构造数据了，用dataset[['Age' , 'Spending Score (1-100)']]提取这两列的所有数据，用.iloc[: , :]保持dataframe格式，再用.values把所有数值提取出来。因为dataframe是key-value格式的嘛，直接抽做不了
## Inertia
Inertia是一个值，用来衡量需要分成几个clusters。从图上可以看出，到4之后进一步提高N对inertia的降低比较小，所以就用4。当然，这个过程要求你先多做几次KMeans，然后调用.inertia_方法得到inertia的值。得到值之后再来作图。
```python
#Segmentation using Age and Spending Score
'''Age and spending Score'''
X1 = dataset[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11): #n is the number of clusters
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

#Selecting N Clusters based the Inertia (Squared Distance between Centroids and data points, should be less)
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

```
得到4是最优的之后，就可以再来一次拟合。当然，k-mean.py里我放的是另两个参数的图，那个取k=5。

之后就是常规操作了，先画个散点图看看长啥样，老一套scatter先把点画出来，KMeans里面储存着labels和cluster_centers_数据，用来画第二个scatter，也就是中心。因为cluster_centers_，第一列是x轴，第二列是y轴，所以就这么画了

其实一般来说做到这里就够了，不需要画分界线

最后的103-120行的代码就照抄吧，它是数据格式的清洗。最后部分的作图和前面一致，故不加赘述。