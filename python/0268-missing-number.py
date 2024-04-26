"""
268 - Missing Number
Hashset | Check existence
Time: O(n) | Space: O(n)

Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
"""

from typing import List

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        num_set = set(nums)

        for i in range(n):
            if i not in num_set:
                return i

        return n