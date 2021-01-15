# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:18:20 2021

@author: billy huang
"""
class ListNode(object):
    def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
# 类似加法的原理, 我们从低位（链条第一位）开始，同位相加，满10高位+1
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        result_tail = result
        carry = 0
                
        while l1 or l2 or carry:            
            val1  = (l1.val if l1 else 0)
            val2  = (l2.val if l2 else 0)
            carry, out = divmod(val1+val2 + carry, 10)    
                      
            result_tail.next = ListNode(out)
            result_tail = result_tail.next                      
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
               
        return result.next