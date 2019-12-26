# 六、字典
Python中字典与Json相同，都是键值对形式：A={‘key’:’Value’,…….}。提取字典中的value使用X[‘key’]进行提取。添加键值对，只需要X[key]=’value’即可。修改只需要X[key]=’value’即可。删除使用del X[‘key’]
为使键值对看上去好看，最后使用如下格式：
 ```
X={
‘key1’=’value1’,
‘key2’=’value2’
}
 ```
调用所有的key只需要使用X.keys()即可。调用所有的value只需要使用X.values()即可
可以把一系列字典嵌套在列表中，只需要分别创建字典，之后X=[a,b,c]即可。同理，把列表嵌套在字典中，只需要把value变成列表即可。
