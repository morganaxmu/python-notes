## 1.testing
first thing you do is writting tests，测试用doctest包

```python
import doctest
def add (x,y):
    """
    >>> add(5,3)
    8
    """
    return x+y
doctest.testmod()
```
上述代码会返回** TestResults(failed=0, attempted=1) ** 如果不正确，就会报错，如下列
```python

9
import doctest
def add (x,y):
    """
    >>> add(5,3)
    9
    """
    return x+y
doctest.testmod()
**********************************************************************
File "__main__", line 4, in __main__.add
Failed example:
    add(5,3)
Expected:
    9
Got:
    8
**********************************************************************
1 items had failures:
   1 of   1 in __main__.add
***Test Failed*** 1 failures.
TestResults(failed=1, attempted=1)
```

## 2.computational complexity
时间复杂度（time complexity）：算法的操作单元数量；最坏情况复杂度：输入n需要的最大操作单元数量；（big-O,f(x)=x^2+x+1=O(x^2)，相当于同阶无穷小的概念（Infinitesimal of the same order）。一般来说讲的时间复杂度就是最坏情况复杂度，举例如下：
```python
def bubblesort(L):
    n = len(L)
    for i in range(0,n-1):
        for j in range(0,n-1):
            if L[j] > L[j+1]:
                temp = L[j]
                L[j] = L[j+1]
                L[j+1] = temp
    return L
```
操作数量：2(n-1)(n-1)，同阶：n^2，时间复杂度O(n^2)

通过复杂度可以衡量算法的效率，像上述sort其实可以每一次只比较n-i-1个，因为每次比较完可以确保这几个的顺序是对的。时间复杂度从小到大:O(k)<O(logn)<O(n)<O(nlogn)<O(n^2)<O(n^3)<O(2^n)<O(n!)