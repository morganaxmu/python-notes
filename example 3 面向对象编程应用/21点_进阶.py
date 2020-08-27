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
            return 'you got 'f'{self.value}'' points! you lose!'
        else:
            return 'you got 'f'{self.value}'' points in total!'
    def score(self):
        return self.value
        
class Bot(Player):
    def __init__(self, name):
        super().__init__(name)   
        self.cards = []
        self.value = 0
        
    def output(self):
        return 'bot got 'f'{self.value}'' points in total!'
   
        
"""初始化"""
deck = Deck()
deck.shuffle()
print("welcome to 21 points game，draw cards from the deck and get as many ponts as you can\n")
print("But if you get over 21 points in total, you will lose the game\n")
print("Notice:AJQK stand for 1,11,12,13 points individually\n")
you = Player('human')
bot = Bot('bot1')
you.get_one(deck.deal())
you.arrange()
print('card in your hand ', end='')
print(you.cards)
"""进行游戏"""
proxy = True
while proxy == True :
    a = str(input("Do you want to draw a card? Y/N "))
    if a == str('Y') or a == str('y'):
        you.get_one(deck.deal())
        you.arrange()
        print('cards in your hand ', end='')
        print(you.cards)
    else:
        proxy = False
print(you.judge())
keep = True
while keep == True:
    bot.get_one(deck.deal())
    bot.arrange()
    if bot.score() < 21:
        continue
    else:
        keep = False
print(bot.output())
if bot.score() > 21:
    if you.score() <= 21:
        print("Congratulations! you win the game!")
        print('cards in your hand ', end='')
        print(you.cards)
        print('cards in bot hand', end='')
        print(bot.cards)
    else:
        print("draw game!")
        print('cards in your hand ', end='')
        print(you.cards)
        print('cards in bot hand', end='')
        print(bot.cards)
else:
    if you.score() > 21:
        print("you lose the game!")
        print('cards in your hand ', end='')
        print(you.cards)
        print('cards in bot hand', end='')
        print(bot.cards)
    else:
        if you.score() >= bot.score():
           print("Congratulations! you win the game!")
           print('cards in your hand ', end='')
           print(you.cards) 
           print('cards in bot hand', end='')
           print(bot.cards)
        else:
           print("you lose the game!")
           print('cards in your hand ', end='')
           print(you.cards) 
           print('cards in bot hand', end='')
           print(bot.cards)