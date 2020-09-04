### 关键字参数
如下列代码所示
```Python
def calc(*args, init_value, op, **kwargs):
    result = init_value
    for arg in args:
        result = op(result, arg)
    for value in kwargs.values():
        result = op(result, value)
    return result
```
在函数的参数列表中可以使用**可变参数**`*args`来接收任意数量的参数，但是其不能接收指定了参数名的参数——例如length=18之类的就不可以

**关键词参数**`**kwargs`会将传入的带参数名的参数组装成一个字典，参数名就是字典中键值对的键，而参数值就是字典中键值对的值

> **注意**：**不带参数名的参数（位置参数）必须出现在带参数名的参数（关键字参数）之前**，否则将会引发异常。

如果希望函数的调用者必须以`参数名=参数值`的方式传参，可以用**命名关键字参数**取代位置参数。所谓命名关键字参数，是在函数的参数列表中，写在`*`之后的参数，代码如下所示。

```Python
def can_form_triangle(*, a, b, c):
    print(f'a = {a}, b = {b}, c = {c}')
    return a + b > c and b + c > a and a + c > b

print(can_form_triangle(c=5, b=4, a=3))
```

> **注意**：上面参数列表中的`*`是一个分隔符，`*`前面的参数都是位置参数，而`*`后面的参数就是命名关键字参数。

### 高阶函数的用法
函数的参数和返回值可以是任意类型的对象，这就意味着**函数本身也可以作为函数的参数或返回值**，这就是所谓的**高阶函数**。
```Python
def calc(*args, init_value, op, **kwargs):
    result = init_value
    for arg in args:
        result = op(result, arg)
    for value in kwargs.values():
        result = op(result, value)
    return result
```
例如上述代码中的`op`就是把函数当作函数的参数，传递进去的必须是一个可用的函数比如`add`，在该代码中`op`代表二元运算函数

`op`参数也可以有默认值，例如我们可以用一个代表加法运算的Lambda函数来作为`op`参数的默认值。
```python
def calc(*args, init_value=0, op=lambda x, y: x + y, **kwargs):
    result = init_value
    for arg in args:
        result = op(result, arg)
    for value in kwargs.values():
        result = op(result, value)
    return result
```

> **注意**：将函数作为参数和调用函数是有显著的区别的，**调用函数需要在函数名后面跟上圆括号，而把函数作为参数时只需要函数名即可**

### Lambda函数
在使用高阶函数的时候，如果作为参数或者返回值的函数本身非常简单，一行代码就能够完成，那么我们可以使用**Lambda函数**来表示。Python中的Lambda函数是没有的名字函数，所以很多人也把它叫做**匿名函数**，匿名函数只能有一行代码，代码中的表达式产生的运算结果就是这个匿名函数的返回值。

定义Lambda函数的关键字是`lambda`，后面跟函数的参数，如果有多个参数用逗号进行分隔；冒号后面的部分就是函数的执行体，通常是一个表达式，表达式的运算结果就是Lambda函数的返回值，不需要写`return` 关键字。

```Python
numbers1 = [35, 12, 8, 99, 60, 52]
numbers2 = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers1)))
print(numbers2)    # [144, 64, 3600, 2704]
```

### 装饰器


