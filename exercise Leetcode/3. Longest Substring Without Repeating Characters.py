# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:47:41 2021

@author: billy huang
"""

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        charset=set()
        l=0
        result=0
        
        for r in range(len(s)):
            while s[r] in charset:
                charset.remove(s[l])
                l+=1
            charset.add(s[r])
            result=max(result,r-l+1)
        return result