# 1. Two Sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

这道题有个坑点，那就是要去重。Input: nums = [2,7,11,15], target = 9的时候，Output: [0,1]，也就是说只认为2+7成立，7+2则不成立。这个其实比较简单，就你做两层循环，第一层的时候i，第二层的时候从j=i+1开始，就可以避免。

我的思路很简单，先建一个空表output用于存放结果，接着两层循环，第一层i从0到len(nums)-1，因为最后一个肯定不需要管；然后第二层从i+1到len(nums)，这样一来得到的是time complexity为O(n^2)的一个玩意儿。

时间复杂度降不下去的，因为你肯定要两层。但是有另一个方法不用再走一个循环，那就是补充。也就是你要a+b=c，直接找c-n在不在这个列表里就可以了。
```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        h = {}
        for i, num in enumerate(nums):
            n = target - num
            if n not in h:
                h[num] = i
            else:
                return [h[n], i]

```
这里用到的enumerate函数，会返回列表的每个元素的rank和value，也就是说，对[a,b,c]会返回0 a;1 b; 2 c;这样的东西。

# 2. Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

Input: l1 = [2,4,3], l2 = [5,6,4]；Output: [7,0,8]；Explanation: 342 + 465 = 807.

这个问题最难的点不在于reverse，而在于它的输入是listnode类型……这玩意儿是Java的……我不懂这玩意儿，所以这道题跳过

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

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
```

# 3. Longest Substring Without Repeating Characters
Given a string s, find the length of the longest substring without repeating characters.Input: s = "abcabcbb";Output: 3;Explanation: The answer is "abc", with the length of 3.

这道题需要用sliding windows的想法。首先创建一个set，逐步读取每个元素放进去，如果遇到重复那就剔除掉之前set里面的这个元素。操作的时候，需要左指针L和右指针R，R直接用循环生成即可，从L=0开始一个一个遍历过去，遇到重复剔除已经放进去的重复的那个然后L+1，那么最长的不重复的长度就是R-L+1.

```python
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
```

# 4. Median of Two Sorted Arrays
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays. Follow up: The overall run time complexity should be O(log (m+n)).

