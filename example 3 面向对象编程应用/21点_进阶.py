# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:05:10 2020

@author: billy huang
"""


from enum import Enum
import random

class Suite(Enum):
    """花色(枚举)"""
    SPADE, HEART, CLUB, DIAMOND = range(4)
    
class Card:
    """牌"""

    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    def __repr__(self):
        suites = '♠♥♣♦'
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        # 根据牌的花色和点数取到对应的字符
        return f'{suites[self.suite.value]}{faces[self.face]}'

    def __lt__(self, other):
        # 花色相同比较点数的大小
        if self.suite == other.suite:
            return self.face < other.face
        # 花色不同比较花色对应的值
        return self.suite.value < other.suite.value
    
    def __int__(self):
        return self.face
    
class Deck:
    """牌堆"""

    def __init__(self):
        # 通过列表的生成式语法创建一个装52张牌的列表
        self.cards = [Card(suite, face) for suite in Suite
                      for face in range(1, 14)]
        # current属性表示发牌的位置
        self.current = 0

    def shuffle(self):
        """洗牌"""
        self.current = 0
        # 通过random模块的shuffle函数实现列表的随机乱序
        random.shuffle(self.cards)

    def deal(self):
        """发牌"""
        card = self.cards[self.current]
        self.current += 1
        return card

    @property
    def has_next(self):
        """还有没有牌可以发"""
        return self.current < len(self.cards)   

        

class Player:
    """玩家"""

    def __init__(self, name):
        self.name = name
        self.cards = []
        self.value = 0
        
    def get_one(self, card):
        """摸牌"""
        self.cards.append(card)
        self.value += int(card)

    def arrange(self):
        self.cards.sort()
    
    def judge(self):
        if self.value > 21:
            return False
        else:
            return self.value
        
class Bot(Player):
    def __init__(self, name):
        super().__init__(name)   
        self.cards = []
        self.value = 0
        
deck = Deck()
deck.shuffle()
"""
a= Player('a')
for i in range(3):
    a.get_one(deck.deal())
print(a.cards)
print(int(a.judge()))
"""
you = Player('a')
a = int(input("初始抽几张牌？"))
for i in range(a):
    you.get_one(deck.deal())
you.arrange()
print('card in your hand', end='')
print(you.cards)
if you.judge() == 0:
    print("you lose!")
else:
    print("you got "+ str(you.judge())+" point in total!")
