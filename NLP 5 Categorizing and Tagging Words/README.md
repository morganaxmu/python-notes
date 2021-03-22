The goal of this chapter is to answer the following questions:

1. What are lexical categories, and how are they used in natural language processing?

2. What is a good Python data structure for storing words and their categories?

3. How can we automatically tag each word of a text with its word class?

tagging is the second step in the typical NLP pipeline, following tokenization

The process of classifying words into their parts-of-speech and labeling them accordingly is known as part-of-speech tagging, POS tagging, or simply tagging. Partsof- speech are also known as word classes or lexical categories. The collection of tags used for a particular task is known as a tagset.

# Using a tagger
使用nltk.ps_tag()即可处理text，该函数会将词划分成不同的class：
```python
import nltk
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
```
其中，CC, a coordinating conjunction；RB, adverbs；IN, a preposition；NN, a noun；JJ, an adjective；present tense verb (VBP)

有时候同一个词会同时拥有两种tag，很正常

.similar() method会寻找同一语境下（也就是axb情况下所有的x）的其他词

# Tagged Corpora
使用nltk.tag.str2tuple()可以把任意表达转换成标准形式，标准形式为tuple(word,tag)
```python
import nltk
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
```

## A Universal Part-of-Speech Tagset
和书上不一样，现在叫Universal

![](5-1.PNG)

```python
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news',  tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.plot(cumulative=True)
```


# Mapping Words to Properties Using Python Dictionaries

dictionary data type (also known as an associative array or hash array in other programming languages)

主要是字典的操作，略过

# Automatic Tagging
The Regular Expression Tagger根据patterns来进行分类，例子见对应py

```python
patterns = [
    (r'.*ing$', 'VBG'), # gerunds
    (r'.*ed$', 'VBD'), # simple past
    (r'.*es$', 'VBZ'), # 3rd singular present
    (r'.*ould$', 'MD'), # modals
    (r'.*\'s$', 'NN$'), # possessive nouns
    (r'.*s$', 'NNS'), # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
    (r'.*', 'NN') # nouns (default)
    ]
regexp_tagger = nltk.RegexpTagger(patterns)
```

The final regular expression «.*» is a catch-all that tags everything as a noun.

LOOKUP tagger会用到UnigramTagger，其可以传递进去一个字典作为模型，根据这个模型来估计。模型size越大，那么准确率就越高。一般来说可以提取已知的某个corpos里面top100之类的来充当模型。

在评估tagger的时候，会用到gold standard test data，这个之后会讲到

# N-Gram Tagging
Unigram taggers are based on a simple statistical algorithm: for each token, assign the tag that is most likely for that particular token.比如frequent大多数时候作为adj使用而非verb所以它就会标注为JJ。UnigramTagger在创建的时候就需要传入tagged sentence作为参数，比如之前的lookup tagger

An n-gram tagger is a generalization of a unigram tagger whose context is the current word together with the part-of-speech tags of the n-1 preceding tokens，简单来说用n-1个的tag和第n个的token进行tagging

2-gram taggers are also called bigram taggers, and 3-gram taggers are called trigram taggers.

可以在tagger的参数里用backoff指定别的tagger，这样当该tagger做不出来的时候就会交给指定的tagger，如：
```python
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print(t2.evaluate(test_sents))

```
保存和读取tagger也很容易
```python
#save tagger
from pickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()
#load tagger
from pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()
```

# Transformation-Based Tagging

230