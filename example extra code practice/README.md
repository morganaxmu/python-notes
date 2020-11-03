## 20201008
这是一节practice课的练习，具体的要求如下：
### Task 1: Duplicate Removal

Write a function that takes as its input a list, and creates a new list with all of the duplicate entries removed. Then print out the new list. You should use a for loop in your solution. 

Try your function on the following lists:

L = [3,7,11,-2,7,-7,1,13,7,2,3,1]

L = ["hi","Rob","Steven","hello","hello world","hi world","world","Rob","hello","hello Steven"]

L = [2,"hi",-2,"2",0,"=","Rob","0",2.0,0]

非常简单，python自带的set（集合）是天然去重的，转换成set再转出来list就可以了

另一种思路则是，构建一个空列表，然后用生成式把空列表里没有的元素给它塞进去

### Task 2: The Guessing Game

Generate a random number from 1 to 99, and ask the user to input a number. Your program should tell the player if their guess is "too high", "too low" or "correct". If the player guesses too high or too low, the program should ask them to input another number, and again tell the player if they are too high, too low or correct. Keep doing this until the player inputs the correct number. When the player inputs the correct number, you should print out a message saying how many guesses the player had.

You will need to remember or look up how to generate a random number in the range 1 to 99, and you will also need some code to ask the user to input a value. You can do this with the code:

x = input("Enter a number:")

A box will appear under your codebox, with the message "Enter a number:". When you enter a number and press enter, that value will be passed to the variable x. 

非常简单的循环，一个while，然后几个if判断条件就结束了

### Task 3: Prime Factorisation

Ask the user for an input number, and output the prime factorisation of that number. E.g. If the user inputs the number 12, the output should be 12 = 2 * 2 * 3. If the user inputs the number 32, the output should be 32 = 2 * 2 * 2 * 2 * 2. If the user enters the number 30, the output should be 30 = 2 * 3 * 5. The user should be able to enter any number up to 100,000,000. Use a line of code similar to that above to ask the user for an input number.

稍微有些麻烦的问题，关键问题在于拆分步骤。首先，要回忆起因式分解我们是怎么做的——x，找到它的一个除数，2、3、5……——除过去，再拿就到的数迭代进去继续找下一个除数。

所以这个问题的解法就是，先构造一个列表用于储存因子，然后开始作法——先试探2是不是，不行就3，不行继续……找到这个因子之后，除掉它，然后继续……直到你最后搞到了1（也就是最后一个因子找到了）

# 练习
## 要求
### 1) Finding the maximum of a list of numbers. [15%]
a) Write a python function to find the maximum (largest element) of a list. Evaluate the performance of your function by timing its execution on a range of list sizes with the python timeit() function, using a randomly generated a list of numbers as input.

b) Create a numpy array based on the same generated data and evaluate the performance of the the numpy amax() function by timing its execution on a range of list sizes.

c) Create a pandas DataFrame based on the same generated data and and evaluate the performance of the the pandas DataFrame.max() function by timing its execution on a range of list sizes.

### 2) Sorting a list of numbers. [10%] 
Using the data structures you created in part 1), compare the performance of the python sort() function, the numpy.sort() function and the pandas DataFrame.sort_values() function on a range of list
sizes.

### 3) Finding the upper quartile of a list of numbers. [10%] 
Write a python function to find the upper quartile of a list, based on the python standard library sorted()
function. Compare the performance of this function with the numpy quantile() function and the pandas DataFrame.quantile() function. To find the upper quartile use the quantile function with a q value of 0.75.

### 4) Testing your code. [5%] 
Write python doctests for the max() and upperQuartile() functions that you have written.

## 思路
### 第一问
第一问其实挺简单的，就是测试时间不好搞。主要原因是jupiter notebook有自带timeit，而其他的IDE嘛，你就得自己写。其实最简单的办法就是用default_timer来计时就好了，即
```python
def timer(func):
    starttime = timeit.default_timer()
    print(func)
    print("The time difference is :", timeit.default_timer() - starttime)
```


### 第二问
其实难点在于这三个方法的调用之前你需要先更改数据类型。从list转换成numpy需要用np.array()函数，然后再调用.sort()方法。而Pandas要麻烦的多，如下所示：
```python
from pandas.core.frame import DataFrame
y = DataFrame(x)
y.columns = ['col1']
```
因为Pandas的DF的sort_values需要一个参数，所以你必须命名一下列。注意命名的时候输入的是一个列表。

### 第三问
应该来说很简单？就第二问的基础上自己写一个函数就是了。唯一的问题就是分位数的计算公式有点扯，那就是1+(n-1)*q来计算对应的项数，然后拆开来，整数部分为n，小数部分为m，最后计算分位数应该是list[n]+(list[n+1]-list[n])*m这种算法

真的好麻烦阿，但是其实挺好搞得