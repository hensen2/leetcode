"""
1 - Two Sum [Easy]
Hash Table | Check Existence | Array
Time: O(n) | Space: O(n)

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            num = nums[i]
            complement = target - num
            if complement in dic:
                return [i, dic[complement]]

            dic[num] = i

        return []
