The goal of this chapter is to answer the following questions:

1. How can we write programs to access text from local files and from the Web, in order to get hold of an unlimited range of language material?

2. How can we split documents up into individual words and punctuation symbols, so we can carry out the same kinds of analysis we did with text corpora in earlier chapters?

3. How can we write programs to produce formatted output and save it in a file?

这章通用：
```python
from __future__ import division
import nltk, re, pprint
```

# Accessing Text from the Web and from Disk
第一步，载入数据。
```python
# for web
from urllib import urlopen
url = "http://www.gutenberg.org/files/2554/2554.txt"
raw = urlopen(url).read()
tokens = nltk.word_tokenize(raw)
# for html
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
raw = nltk.clean_html(html)
tokens = nltk.word_tokenize(raw)
# for RSS, you need feedparser
# for local files
f = open('document.txt')
raw = f.read()
```
清洗数据的第一步是tokenization，会生成a list of words and punctuation，用nltk自带的即可，如上所示

# Strings: Text Processing at the Lowest Level
字符串的基础操作，比较重要的有：
```
s.find(t) - Index of first instance of string t inside s (-1 if not found)
s.rfind(t) - Index of last instance of string t inside s (-1 if not found)
s.index(t) - Like s.find(t), except it raises ValueError if not found
s.rindex(t) - Like s.rfind(t), except it raises ValueError if not found
s.join(text) - Combine the words of the text into a string using s as the glue
s.split(t) - Split s into a list wherever a t is found (whitespace by default)
s.splitlines() - Split s into a list of strings, one per line
s.lower() - A lowercased version of the string s
s.upper() - An uppercased version of the string s
s.titlecase() - A titlecased version of the string s
s.strip() - A copy of s without leading or trailing whitespace
s.replace(t, u) - Replace instances of t with u inside s
```

# Text Processing with Unicode
对Unicode的解释

# Regular Expressions for Detecting Word Patterns
常用的表达为^.$?这四个符号。

The caret symbol ^ matches the start of a string, the $ matches the end, The . wildcard symbol matches any single character. the ? symbol specifies that the previous character is optional. Thus «^e-?mail
$» will match both email and e-mail.

+ simply means “one or more instances of the preceding item”, *, which means “zero or more instances of the preceding item.”

ed|ing|s Matches one of the specified strings (disjunction)，也就是or。

想搜索多个的时候，可以在[]内填上模糊要求，如下所示：
```
>>> [w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]
['gold', 'golf', 'hold', 'hole']
```
它搜索的就是：^开头为[ghi]中任意一个，第二个为[mno]中任意一个，第三个为[jlk]中任意一个，$结尾为[def]中的任意一个

用（）包起来虽然可以让你在前面和后面加东西，但是()parentheses会限制操作的范围，要提取全部的话就要在()开头加?:才可以。如果你想拆分split the word into stem and suffix，那就需要给前面一部分也打上括号，如下所示：
```python
word = ['liked','go','going','goes']
print([w for w in word if re.search('ed$|ing$|s$',w)])
print(re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
# result ['ing']
print(re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
# result ['processing']
print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
# result [('process', 'ing')]
```
用*来处理因为greedy会找尽可能多的match的，所以有的时候需要加个？
```python
print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes'))
# [('processe', 's')]
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes'))
# [('process', 'es')]
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language'))
# [('language', '')]
```

用<>包住则会搜索特定词，比如"<a> <man>" finds all instances of a man in the text，可以搭配上()来搜索两个特定词中间的词（因为括号会提取这一部分出来）
```
>>> moby.findall(r"<a> (<.*>) <man>")
monied; nervous; dangerous; white; white; white; pious; queer; good;
mature; white; Cape; great; wise; wise; butterless; white; fiendish;
pale; furious; better; certain; complete; dismasted; younger; brave;
brave; brave; brave
```

用{}回表示重复的次数，{n}重复n次，{n,}至少n次，{,n}不超过n次，{n,m}你懂的。比如下面的例子表示sequences of three or more words starting with the letter l
```
>>> chat.findall(r"<l.*>{3,}")
lol lol lol; lmao lol lol; lol lol lol; la la la la la; la la la; la
la la; lovely lol lol love; lol lol lol.; la la la; la la la
```

另一个更常用的是r，会停止一切转义

# Useful Applications of Regular Expressions
The re.findall() (“find all”) method finds all (non-overlapping) matches of the given regular expression.

这个主要讲应用，它把consonant-vowel（CV）序列，也就是辅音的序列比如ka,si之类的抽出来，用CFD来看统计，发现ti是0si是100，那就说明所有的ti都读si

当你的列表的元素是(index,word)格式的时候，可以用nltk.Index将index转换成列表的index，这样就能根据index进行提取，把所有包含这个的全部提取出来
```python
cv_word_pairs = [(cv, w) for w in rotokas_words
                 for cv in re.findall(r'[ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
print(cv_index['su'],cv_index['po'])
```

# Normalizing Text
```python
set(w.lower() for w in text)
```
如上所示，用lower转换成小写的过程就是一个normalization的过程。

有的时候会用到stemmer以去掉同一个词的词尾affixes（也就是说把lying转换回lie），最常用的nltk.PorterStemmer()。而WordNet lemmatizer也会去掉词尾，但是它只会处理已经在字典里面的（比如里面有women和woman，那就会处理women；里面有lying没有lie就不会处理lying）The WordNet lemmatizer is a good choice if you want to compile the vocabulary of some texts and want a list of valid lemmas (or lexicon headwords).

```python
# lemmatizer
wnl = nltk.WordNetLemmatizer()
tokens = [wnl.lemmatize(t) for t in tokens]

```

# Regular Expressions for Tokenizing Text
Tokenization is the task of cutting a string into identifiable linguistic units that constitute a piece of language data.

首先要split，用.split()方法，一般参数如下所示
```
\b - Word boundary (zero width)
\d - Any decimal digit (equivalent to [0-9])
\D - Any non-digit character (equivalent to [^0-9])
\s - Any whitespace character (equivalent to [ \t\n\r\f\v]
\S - Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
\w - Any alphanumeric character (equivalent to [a-zA-Z0-9_])
\W - Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])
\t - The tab character
\n - The newline character
```
133页这里有一段代码的output和它给的不一样，不清楚是什么原因。

# Segmentation
Tokenization is an instance of a more general problem of segmentation.有些语言（比如中文）there is no visual representation of word boundaries，而英文中如果去掉space你一时半会也不一定能看出来
```python
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
print(segment(text, seg1),segment(text, seg2))
```
用上示的代码就可以把segmentation的问题转化成find the bit string that causes the text string to be correctly segmented into words.通过定义一个objective function（lexicon score = sum(len(lexicon)+1), deivation score = sum(len(derivation))，两个相加得到total score)，通过randomly perturb the zeros and ones proportional to the “temperature”; with each iteration the temperature is lowered and the perturbation of boundaries is reduced.(通过引入随机扰动，也就是和“温度”成正比的零和一； 每次迭代都会降低温度，并减少边界的扰动。)最后得到分数最低的就是最佳的了

具体代码示例可以看对应python文件

# Formatting: From Lists to Strings
一般使用.join()方法，.前面跟着的是“胶水”，也就是你list的element之间用啥玩意儿黏在一起，一般就是whitespace
```
>>> silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
>>> ' '.join(silly)
'We called him Tortoise because he taught us .'
>>> ';'.join(silly)
'We;called;him;Tortoise;because;he;taught;us;.'
>>> ''.join(silly)
'WecalledhimTortoisebecausehetaughtus.'
```
string formatting expressions，其实就是格式化输出，和.format一样
```python
import nltk
fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
for word in fdist:
    print('%s->%d;' % (word, fdist[word]))
```
The %s and %d symbols are called conversion specifiers. They start with the % character and end with a conversion character such as s (for string) or d (for decimal integer). The string containing conversion specifiers is called a format string.

设定间隔也很简单，%6s就是往左空6格producing a string that is padded to width 6; %-6s就是右边空6格；如果不知道的话可以用*来代替长度，然后后面第一个填长度的变量
```
>>> '%6s' % 'dog'
'      dog'
>>> '%-6s' % 'dog'
'dog      '
>>> width = 6
>>> '%-*s' % (width, 'dog')
'dog      '
``
保存文件，很简单
```python
output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    output_file.write(word + "\n")
```
