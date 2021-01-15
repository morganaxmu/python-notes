# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:52:21 2021

@author: billy huang
"""

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        output=[]
        for i in range(len(nums)-1):
            a1=nums[i]
            for j in range(i+1,len(nums)):
                a2=nums[j]
                if a1+a2==target:
                    output.append(i)
                    output.append(j)
        return output