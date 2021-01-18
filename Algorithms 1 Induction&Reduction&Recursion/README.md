# introduction
Reduction means transforming one problem to another. We normally reduce an unknown problem to one we know how to solve. The reduction may involve transforming both the input (so it works with the new problem) and the output (so it’s valid for the original problem).

Reduction字面意思是“减少”，此处的意思更贴近“知识迁移”。也就是说把一个未知问题转换成已知的问题来解决。因为是迁移，所以input和output可能要做变换，这个比较接近于数学。

Induction (or, mathematical induction) is used to show that a statement is true for a large class of objects (often the natural numbers). We do this by first showing it to be true for a base case (such as the number 1) and then showing that it “carries over” from one object to the next (if it’s true for n–1, then it’s true for n).

Induction“归纳”，先从比较小的case开始，接着推广到general。

Recursion is what happens when a function calls itself. Here we need to make sure the function works correctly for a (nonrecursive) base case and that it combines results from the recursive calls into a valid solution.

Recursion“递归”。

# Oh, That's Easy!
首先来看第一个例子，从一个List的数里面找出最接近的两个nonidentical的数
```python
from random import randrange
seq = [randrange(10**10) for i in range(100)]
dd = float("inf") #这玩意儿是无穷大
for x in seq:
	for y in seq:
		if x == y: continue
		d = abs(x-y)
		 if d < dd:
			 xx, yy, dd = x, y, d

```
这玩意儿是一个quadratic的，也就是O(n^2)的，所以要想办法简化一下。

>>> 要点1：sequences（序列）在sorted（排序好了）的情况下更容易被处理，sorting是一个loglinear也就是O(nlgn)的算法。

运用这个思想，我们可以把这个序列排序了，这样一来最接近的数字必然是挨在一起的，就可以少写一个循环了
```python
seq.sort()
dd = float("inf")
for i in range(len(seq)-1):
	x, y = seq[i], seq[i+1]
	if x == y: continue
	d = abs(x-y)
	if d < dd:
		xx, yy, dd = x, y, d
```
Reduction迁移就相当于把问题A替换为等价的问题B，只要你能解决B你就能解决A

# one, two ,many
数学归纳法相关内容，略过不加赘述。inductive step指的是数学归纳法里从n-1到n这一步，假设n-1成立被称为the inductive hypothesis。

其中一个很有意思的想法就是拆分问题，把一个问题拆成subquestion。例如：一块8X8但是缺了一角（1X1）的棋盘要如何用L形状（2+1的L，不是3+1）来覆盖？拆分的方法就是把这个棋盘分为四块，然后把三块没有缺的棋盘中间挖掉一个L，这样的话就得到了和初始的棋盘形状类似的四块子棋盘——相当于把N（8）问题拆成了N（4）*4+一个L。

# Mirror, Mirror
之前讲到拆分问题的时候，用了棋盘问题，这个就可以用recursion也就是迭代来解决。
```python
def cover(board, lab=1, top=0, left=0, side=None):
    if side is None: side = len(board)
    # Side length of subboard:
    s = side // 2
    # Offsets for outer/inner squares of subboards:
    offsets = (0, -1), (side-1, 0)
    for dy_outer, dy_inner in offsets:
        for dx_outer, dx_inner in offsets:
            # If the outer corner is not set...
            if not board[top+dy_outer][left+dx_outer]:
            # ... label the inner corner:
                board[top+s+dy_inner][left+s+dx_inner] = lab
    # Next label:
    lab += 1
    if s > 1:
        for dy in [0, s]:
            for dx in [0, s]:
            # Recursive calls, if s is at least 2:
                lab = cover(board, lab, top+dy, left+dx, s)
                # Return the next available label:
    return lab

board = [[0]*8 for i in range(8)] # Eight by eight checkerboard
board[7][7] = -1 # Missing corner
cover(board)
for row in board:
    print((" %2i"*8) % tuple(row))
```
这个代码的思路比较简单，首先确认这个棋盘哪三个角是有完整的，也就是检查(0,0)(0,side-1)(side-1,0)(side-1,side-1)的位置。找到完整的之后，根据之前的思想分为四块，然后把角完整的三块中心的，也就是(s-1,s-1)(s-1,s)(s,s-1)(s,s)四个中对应角完整位置的给标记上，这样就取出了第一个L，并拆分成四个和之前形状完全一样的checkerboard。

之后就是递归了，再来一次。注意这里的s必须要用向下取整的整除，而不是单纯的除。

任何递归基本都能写成iteration（迭代）。对于排序sorting问题，有两张迭代的思路：一种就是假设前n-1已经排序好了，把第n个放到它应该在的位置(insertion sort)，第二种思路就是找到没有排序的里面最大的然后放到n的位置(selection sort)。

# Designing with Induction (and Recursion)
## Finding a Maximum Permutation
假设现在电影院里面有8个人，他们的座位是随机分配的，因为他们都想要自己最喜欢的座位并且不是最喜欢的那就拒绝换座，所以现在你要找出一种座位分配让他们最开心。

这个例子是一个很典型的bipartite graph（二部图），也就是可以把nodes分成两个set，相邻的node可以标注上不同的颜色。

而reduction这个问题的思路很简单，先从induction开始：我们假设n-1个人已经排列好了是最优的，那么现在考虑多加一个人进来；如果这个人最喜欢的位置和别人重了，那么n-1的排列依然是最优的，如果他喜欢一个新的，那么就加一个新的给他。

换言之，如果有座位没人要，那么把这个座位和已经坐在上面的人删掉是不会影响最优排列的（因为这意味着坐在上面的这个人和别人撞最喜欢了）。所以reduction这个问题的思路就是——先找出没人要的位置，然后把坐在上面的人和这个位置删除掉。等所有位置都有人要了，那每个人都得到自己想要的位置就是最优的了。

```python
def naive_max_perm(M, A=None):
    if A is None: # The elt. set not supplied?
        A = set(range(len(M))) # A = {0, 1, ... , n-1}
    if len(A) == 1: return A # Base case -- single-elt. A
    B = set(M[i] for i in A) # The "pointed to" elements
    C = A - B # "Not pointed to" elements
    if C: # Any useless elements?
        A.remove(C.pop()) # Remove one of them
        return naive_max_perm(M, A) # Solve remaining problem
    return A # All useful -- return all

M = [2, 2, 0, 5, 3, 5, 7, 4]
print(naive_max_perm(M))
```
这个方法的缺点是每次都要生成一次B，导致是O(n^2)，所以要进一步简化的话就用一个列表来保存，然后修改里面的数据就好了。也就是说，这次我们不是每次都生成一个列表来保存有人选择的座位，而是用count代替，储存每个座位被指名的数量，如果指名数变成0那就意味着可以删掉了；删人的时候删掉其指名，对应count-1，就直接修改count列表里的值，而不用每次都生成一个列表
```python
def max_perm(M):
    n = len(M) # How many elements?
    A = set(range(n)) # A = {0, 1, ... , n-1}
    count = [0]*n # C[i] == 0 for i in A
    for i in M: # All that are "pointed to"
        count[i] += 1 # Increment "point count"
    Q = [i for i in A if count[i] == 0] # Useless elements
    while Q: # While useless elts. left...
        i = Q.pop() # Get one
        A.remove(i) # Remove it
        j = M[i] # Who's it pointing to?
        count[j] -= 1 # Not anymore...
        if count[j] == 0: # Is j useless now?
            Q.append(j) # Then deal w/it next
    return A # Return useful elts.

print(max_perm(M))
```

>>> Idea：Count是一个非常有用的工具，相当于你有一个列表，你的对象就是列表的index，然后count就是value。这个可以用来排序，counting sort的例子如下，它可以达到O(n)级别快很多

```python
from collections import defaultdict #使用这个函数能生成类似[(key,value),(key,value)]格式的列表
def counting_sort(A, key=lambda x: x):
    B, C = [], defaultdict(list) # Output and "counts"
    for x in A:
        C[key(x)].append(x) # "Count" key(x)
    for k in range(min(C), max(C)+1): # For every key in the range
        B.extend(C[k]) # Add values in sorted order
    return B
```
这里解释一下，首先生成了一个类似字典的列表，然后第一个循环把未排序的列表的所有元素以key-value的形式储存进了这个列表。也就是说，对于需要排序的数字，那么就令对于key的value为这个数字。比如排序[3,2,5]，那么就会得到[[2,2],[3,3],[4,][5,5]]这么一个相当于已经排序好了的东西。.extend方法会在末尾追加，而对类似字典的列表C来说，min和max会取已经有的key里面的最大最小的，这样就可以用线性的算法生成排序好的列表。

## The Celebrity Problem
Celebrity，也就是名人，特征是虽然他不认识其他人但是大家都认识他。写程序实现那就是一个一个遍历过去，看看是不是每个人都认识他并且他不认识每个人。可以用一个列表储存一系列列表，每个人都有一个列表用于储存他认识的人。简单的解法如下：

```python
def naive_celeb(G):
    n = len(G)
    for u in range(n): # For every candidate...
        for v in range(n): # For everyone else...
            if u == v: continue # Same person? Skip.
            if G[u][v]: break # Candidate knows other
            if not G[v][u]: break # Other doesn't know candidate
        else:
            return u # No breaks? Celebrity!
    return None # Couldn't find anyone

from random import randrange
n = 100
G = [[randrange(2) for i in range(n)] for i in range(n)]
c = randrange(n)
for i in range(n):
    G[i][c] = True
    G[c][i] = False
print(naive_celeb(G))
```
很明显这个是quadratic的，还可以改进算法，
```python
def celeb(G):
    n = len(G)
    u, v = 0, 1 # The first two
    for c in range(2,n+1): # Others to check
        if G[u][v]: u = c # u knows v? Replace u
        else: v = c # Otherwise, replace v
    if u == n: c = v # u was replaced last; use v
    else: c = u # Otherwise, u is a candidate
    for v in range(n): # For everyone else...
        if c == v: continue # Same person? Skip.
        if G[c][v]: break # Candidate knows other
        if not G[v][c]: break # Other doesn't know candidate
    else:
        return c # No breaks? Celebrity!
    return None # Couldn't find anyone
```
首先，给定一个初始的u和v，如果G[u][v]为1，说明u认识v，那么就替换掉u，看看别人是不是认识v;如果为0，那么说明u不认识v，v就不可能是名人，那就替换掉v，继续下一个候选人c。第二步，检查我们找到的这个c是不是名人，如果u=n那就说明到最后一个了，根据前述推理这说明u不会是名人，在接下来的步骤里就用v代替c；反之，用u代替c。接着进行检查，这个c是不是满足名人的条件，如果是那就return。

## Topological Sorting
finding an ordering that respect the dependencies (so that all the edges point forward in the ordering) is called topological sorting. directed acyclic graph (DAG)是只要用箭头表示dependencies的图都可以，但是topological sorting必须箭头都向右。

就像安装软件一样，如果你缺了必须的库/环境那肯定不行，这种安装顺序就是topologically sorted order

这里的代码因为没给input所以是没办法输入的（我也不知道输入该长啥样），第一种思路就是，假设L是存放结果的列表，先找到那些入度为零的节点，把这些节点放到L中，因为这些节点没有任何的父节点。然后把与这些节点相连的边从图中去掉，再寻找图中的入度为零的节点。对于新找到的这些入度为零的节点来说，他们的父节点已经都在L中了，所以也可以放入L。重复上述操作，直到找不到入度为零的节点。如果此时L中的元素个数和节点总数相同，说明排序完成；如果L中的元素个数和节点总数不同，说明原图中存在环，无法进行拓扑排序。

也就是说，我们只要找到一个入度为0的节点，我们就能移除它因为它一定不依赖于其他节点。所以化简的办法就是counting。

# Stronger Assumptions
