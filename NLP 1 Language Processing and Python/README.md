# nltk
对于text的nltk.text.Text类型，使用.concordance精准search，会返回所有包含的词句，即A concordance permits us to see words in context；使用.similar寻找在相似的上下文环境内出现的词，即What other words appear in a similar range of contexts；使用.common_contexts来寻找多个词shared的contexts，即examine just the contexts that are shared by two or more words；使用.dispersion_plot会显示一个图告诉你how many words from the beginning it appears，实际上就是一个这个词出现的分布图；.generate()可以随机生成一段类似文风的文字；使用.count('text')可统计某词出现的次数

使用FreqDist()函数来生成一个frequency distribution，它储存了每个字的出现频率；对该对象使用.plot(n,cumulative=True)可以生成一个图；A collocation is a sequence of words that occur together unusually often.使用.collocation就可以找到这些词；bigrams()可以抽出一个list里面相近的词组成pair，比如['a','b','c']会组出ab,bc；

# 语言问题
人可以自动认知语义避免歧义，we automatically disambiguate words using context, exploiting the simple fact that nearby words have closely related meanings.

anaphora resolution—identifying what a pronoun or noun phrase refers to，也就是代词指的到底是哪一个。举个例子，The thieves stole the paintings. They were subsequently sold/caught/found，they到底指画还是贼？translation的时候就会遇到这个问题（比如英语变法语）

再比如说，texts能否support某一个hypothesis的问题，也是一个问题