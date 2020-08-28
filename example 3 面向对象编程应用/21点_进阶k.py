# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:05:10 2020

@author: billy huang

@modifed by dcmk
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
        suites = "♠♥♣♦"
        faces = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        # 根据牌的花色和点数取到对应的字符
        return f"{suites[self.suite.value]}{faces[self.face]}"

    def __lt__(self, other):
        # 花色相同比较点数的大小
        if self.suite == other.suite:
            return self.face < other.face
        # 花色不同比较花色对应的值
        return self.suite.value < other.suite.value
    
    def __int__(self):
        return self.face + 1
    
class Deck:
    """牌堆"""
    def __init__(self):
        # 通过列表的生成式语法创建一个装52张牌的列表
        self.cards = [Card(suite, face) for suite in Suite
                      for face in range(13)]
        self.__shuffle()

    def __shuffle(self):
        """洗牌"""
        random.shuffle(self.cards)
        # current属性表示发牌的位置
        self.current = 0

    def deal(self):
        """发牌"""
        for card in self.cards:
            yield card
    
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

    def getValue(self):
        return self.value
        
def result(msg):
    print(msg)
    print("cards in your hand", you.cards)
    print("cards in bot hand", bot.cards)


if __name__ == "__main__":
    deck = Deck().deal()
    print("Welcome to 21 points game，draw cards from the deck and get as many ponts as you can\n")
    print("But if you get over 21 points in total, you will lose the game\n")
    print("Notice: AJQK stand for 1,11,12,13 points individually\n")
    you = Player("human")
    bot = Player("bot1")
    you.get_one(next(deck))
    you.arrange()
    print("Card in your hand", end="")
    print(you.cards)
    """进行游戏"""
    while True:
        a = str(input("Do you want to draw a card? Y/N "))
        if a.upper() == "Y":
            you.get_one(next(deck))
            you.arrange()
            print("Cards in your hand", you.cards)
        else:
            break
    print(f"you got {you.getValue()} point in total!")
    while True:
        bot.get_one(next(deck))
        bot.arrange()
        if bot.getValue() >= 21:
            break
    print(f"bot got {bot.getValue()} point in total!")
    if bot.getValue() <= you.getValue() <= 21 or you.getValue() <= 21 < bot.getValue():
        result("Congratulations! You win the game!")
    elif bot.getValue() > 21 and you.getValue() > 21:
        result("Draw game!")
    else:
        result("You lose the game!")
